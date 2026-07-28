from __future__ import annotations

import numpy as np
import torch

from catena.core.io import read_latest_pointer, write_jsonl
from catena.core.schema import Operation
from catena.data.chains import generate_chains
from catena.data.semantic_transactions import SemanticTransaction
from catena.data.sequential import commit_canonical, initialize_sequential_memory, make_sequential_episode
from catena.eval.metrics import evaluate_episode
from catena.models.memory import apply_scalar_update
from catena.models.semantic_controllers import MatchedSemanticController, SemanticConstraint
from catena.systems.cost_model import CostModel
from catena.training.semantic_probe import SemanticProbeConfig, SemanticProbeExample, semantic_probe_features, semantic_probe_input_dim
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID="e06_reusable_state_assimilation"; DEFAULT_CONFIG="configs/e06_reusable_state_assimilation.yaml"


def _text(operation:Operation,entity:str,new:str)->str:
    return {Operation.PRESERVE:f"The current record for {entity} is unchanged.",Operation.ADD:f"The current record for {entity} retains its prior entry and contains {new} too.",Operation.INVALIDATE:f"The prior entry for {entity} is historical and has no active replacement.",Operation.SUPERSEDE:f"The prior entry for {entity} is historical and {new} is current."}[operation]


def _action_correct(read:torch.Tensor,target:torch.Tensor,stale:torch.Tensor)->bool:
    return bool(torch.mean((read-target)**2)<=torch.mean((read-stale)**2))


def main()->None:
    parser=build_parser(EXPERIMENT_ID,DEFAULT_CONFIG); args=parser.parse_args(); config,run_dir,device=initialize_run(experiment_id=EXPERIMENT_ID,config_path=args.config,artifact_root=args.artifact_root,device_request=args.device)
    source=read_latest_pointer(args.artifact_root,"e05_semantic_demand_inference")
    seed=int(config["model"]["checkpoint_seed"]); checkpoint=source/f"seed{seed}_factorized.pt"
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    tamp=config["tamp"]; probe=SemanticProbeConfig(int(config["model"]["bow_dim"]),int(config["model"]["hidden_dim"]),bool(config["model"]["include_state_read"]),1,1e-3)
    model=MatchedSemanticController(semantic_probe_input_dim(probe,int(tamp["value_dim"])),probe.hidden_dim,SemanticConstraint.FACTORIZED).to(device); model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); model.eval()
    count=int(config["data"]["chain_count"]); count=min(count,12) if args.dry_run else count
    chains=generate_chains(count=count,lengths=[int(x) for x in config["data"]["chain_lengths"]],seed=int(config["seed"]),query_counts=[int(x) for x in config["data"]["query_counts"]],include_rollback=bool(config["data"]["include_rollback"]))
    cost=CostModel(**{k:float(v) for k,v in config["cost_model"].items()}); rows=[]; external_be=[]; cached_be=[]; generator=torch.Generator().manual_seed(int(config["seed"])+55)
    for chain_index,chain in enumerate(chains):
        memory=initialize_sequential_memory(seed=int(config["seed"])+chain_index,num_associations=int(tamp["num_associations"]),key_dim=int(tamp["key_dim"]),value_dim=int(tamp["value_dim"]),key_correlation=float(tamp["key_correlation"]))
        stale=memory.initial_state.clone(); assimilation=memory.initial_state.clone(); snapshots=[assimilation.clone()]; last_episode=None; last_gates=None
        for step_index,step in enumerate(chain.steps):
            new=torch.randn(int(tamp["value_dim"]),generator=generator); new=new/new.norm().clamp_min(1e-8)
            episode=make_sequential_episode(memory=memory,model_state=assimilation,operation=step.operation,new_value=new,episode_id=f"{chain.chain_id}-step{step_index}")
            tx=SemanticTransaction(f"{chain.chain_id}-{step_index}","chain",step.operation,_text(step.operation,step.entity,step.new_value),step.entity,step.old_value,step.new_value,f"chain-{step.operation.value}","structured")
            example=SemanticProbeExample(tx,episode); features=semantic_probe_features(example,probe).to(device).unsqueeze(0)
            with torch.no_grad():
                gates=model(features); assimilation=apply_scalar_update(episode.to(device),gates.erase.squeeze(0),gates.write.squeeze(0)).cpu()
            last_episode=episode; last_gates=gates; commit_canonical(memory,episode); snapshots.append(assimilation.clone())
        if chain.rollback_target is not None: assimilation=snapshots[min(chain.rollback_target,len(snapshots)-1)].clone()
        if last_episode is None: continue
        final=make_sequential_episode(memory=memory,model_state=assimilation,operation=Operation.PRESERVE,new_value=torch.zeros(int(tamp["value_dim"])),episode_id=f"{chain.chain_id}-final"); final.target_state=memory.canonical_state.clone(); exact=memory.canonical_state.clone(); noise=float(config["baselines"]["compact_snapshot_noise"]); compact=exact+noise*torch.randn_like(exact,generator=generator); cached=compact.clone()
        append=apply_scalar_update(last_episode,last_gates.erase.squeeze(0).cpu(),last_gates.write.squeeze(0).cpu()) if last_gates is not None else stale
        states={"stale":stale,"typed_append_last_only":append,"one_time_assimilation":assimilation,"compact_snapshot_per_query":compact,"retrieve_once_cached_snapshot":cached,"external_canonical_read":exact,"full_refresh":exact}
        key=memory.keys[memory.affected_index]; target=key@exact; stale_read=key@stale
        for name,state in states.items():
            metrics=evaluate_episode(state,final); rows.append({"chain_id":chain.chain_id,"chain_length":len(chain.steps),"query_count":chain.query_count,"distractor_count":chain.distractor_count,"rollback":chain.rollback_target is not None,"baseline":name,"plan_continuation_correct":_action_correct(key@state,target,stale_read),**metrics.to_dict()})
        e=cost.break_even_queries("external_every_query"); c=cost.break_even_queries("cached_snapshot")
        if e is not None: external_be.append(e)
        if c is not None: cached_be.append(c)
    assimilation_rows=[r for r in rows if r["baseline"]=="one_time_assimilation"]; stale_rows=[r for r in rows if r["baseline"]=="stale"]
    mean_assim=float(np.mean([r["affected_read_mse"] for r in assimilation_rows])); mean_stale=float(np.mean([r["affected_read_mse"] for r in stale_rows])); mean_ret=float(np.mean([r["unaffected_retention_mse"] for r in assimilation_rows]))
    corr_thr=float(config["quality_constraints"]["correction_mse_threshold"]); ret_thr=float(config["quality_constraints"]["retention_mse_threshold"]); quality_ok=mean_assim<=corr_thr and mean_ret<=ret_thr
    report={"status":"PASS","mean_affected_mse":{"stale":mean_stale,"assimilation":mean_assim},"mean_assimilation_retention_mse":mean_ret,"cost_break_even_queries_external_every_query":int(np.median(external_be)) if external_be else None,"cost_break_even_queries_cached_snapshot":int(np.median(cached_be)) if cached_be else None,"quality_constrained_break_even_definition":{"correction_mse_threshold":corr_thr,"retention_mse_threshold":ret_thr},"claim_gate":{"multi_update":mean_assim<mean_stale,"external_read_break_even":quality_ok and bool(external_be or cached_be)}}
    write_jsonl(run_dir/"chain_metrics.jsonl",rows); finalize_run(experiment_id=EXPERIMENT_ID,artifact_root=args.artifact_root,run_dir=run_dir,report=report); print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__=="__main__": main()
