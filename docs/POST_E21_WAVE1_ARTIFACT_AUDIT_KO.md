# CATENA Post-E21 Wave 1 Artifact 감사

감사 기준:

- Repository: `/home/minjun_dev/CATENA`
- Base MAIN source lock:
  `51156242dfc429cb66d577c144b8d38a5ae38551`
  (`post-e21-main-source-lock-20260728T174131Z`)
- Sharded confirmatory source lock:
  `1ef51cdd488b411d51a3dac85eaa4fc8d5b04d1b`
  (`post-e21-sharded-source-lock-20260729T122800Z`)
- Artifact root: `/data/minjun_dev/CATENA/artifacts`
- Workspace `artifacts` realpath:
  `/data/minjun_dev/CATENA/artifacts`

각 canonical run에서 `run_manifest.json`, `protocol_lock.json`,
`data_manifest.json`, `report.json`, raw/seed rows, checkpoint map,
`latest.json`, non-finite/duplicate 여부와 dependency SHA를 확인했다.
Scientific disposition은 report에서 그대로 읽었으며 감사 과정에서
threshold나 metric을 재계산해 바꾸지 않았다.

## Per-run provenance

| Exp | Source commit/lock | Source fingerprint | Config SHA-256 | Data SHA-256 | Checkpoint aggregate |
|---|---|---|---|---|---|
| E22a | `51156242dfc429cb66d577c144b8d38a5ae38551` | `9f97b8cbdae1ed1abec7784b79893dea860bda535a1942188161d6ab5875e4ac` | `6a40aae4798a2e86c7d4f4ccfe2b50997fc01f27f773a539635b7d27fd7178d4` | `1031b969f58984133055b8f45e48bb2d41f74998c100a5b866a3e8c6fed8bd56` | `3fa1378327b540ddce9b10e6d7643718cf0c4e626b9f8b7f01214d5388414f77` |
| E22b | `1ef51cdd488b411d51a3dac85eaa4fc8d5b04d1b` | `86e30646577a5bd18a608d060a03cd98d3de6b15dee83a702d6b8d7a37571683` | `49c719dcf93ae8b6ecd5bd6c5c7b91bf8c855b99eb0348799b86fc5da73d724f` | `c89d7239635c6d82da28d137f463dc7d733d96c9856f052c703cf06af5240f00` | `f940b2de69ef2d832bcd28c4ab0855124c177b0b433b9ce3d684efe16da1a97c` |
| E23a | `51156242dfc429cb66d577c144b8d38a5ae38551` | `9f97b8cbdae1ed1abec7784b79893dea860bda535a1942188161d6ab5875e4ac` | `214e68bdf8aa13056ebab45aad8a9705062fec6273c533efe00088e8582d4782` | `15cbaa99ac115688e0524a03e764a54d3ae8ccf5551f932c677d256c0a360486` | `79f2b049a2c0950f333ec497905b2f377c9482f5df1e7418b8aeac81bfd9ac6c` |
| E23b | `1ef51cdd488b411d51a3dac85eaa4fc8d5b04d1b` | `86e30646577a5bd18a608d060a03cd98d3de6b15dee83a702d6b8d7a37571683` | `b2008d2323a7352e8ddc44d6c6eaec27f6752dce08ef6c5a673432903730b0cd` | `bb4cfb44ddc407837f9a29b6e377e31946ea7ffcd4fb3bb4662478e453d3859a` | `42f54a7e09ad6c8954289a4d82f9b82087fdd48b6f243ce3182df35039b06d3a` |
| E24a | `51156242dfc429cb66d577c144b8d38a5ae38551` | `9f97b8cbdae1ed1abec7784b79893dea860bda535a1942188161d6ab5875e4ac` | `3277eb5bd6539ec00a2a4c9d665beef5d16daab49062637d08918e3a6faf1ab6` | `1c555f7719917bf9935b01e7868ee616c453d5811eee39bcd2ead06e55324652` | `ba53d4e170f5812abd590b9e256800d880be5a653e80668266116f4c3ca648a1` |
| E24b | `51156242dfc429cb66d577c144b8d38a5ae38551` | `9f97b8cbdae1ed1abec7784b79893dea860bda535a1942188161d6ab5875e4ac` | `0461fe1d483358224382ab4963f86de986f87e8f9980ec3daad349be580cded9` | `75cf9e33738d40c0595a59b76e993a02d7608badcbb8b150b3af744dfef5a46f` | `75f3d930f3d6e8bf5ff5f2448b970597e255736e81d883e0d34928b9599a8daf` |
| E25a | Internal commit not recorded; official commits pinned | `37ef731d5a1bbb288756e71f8d011227d15746007e1ba6fc24b9f35dcb5a6c97` | `c35bb69c8fa71f667933b6c5f1592bd9deafbf5fd1ad80077ff35b78b144fdd6` | `4957eb4a78ba5b6cfa65ec3d9d3c9e3dfdf8b03bbef177f12f408f0a2f0d47a9` | N/A |
| E25b | `51156242dfc429cb66d577c144b8d38a5ae38551` | `9f97b8cbdae1ed1abec7784b79893dea860bda535a1942188161d6ab5875e4ac` | `263aadf92efa562cc2ae8d8eae1c1eb637a29d0ddb687ec7d7b262ca68740ff0` | `e753b14e6f641d21ab4b3825db4892ed10dad0773f6ea8f6b1deb18f00bac437` | N/A |

E25a official GDN2/FLA commits는 각각
`95709fc250357c2dd109361c353192f2aa5913f9`,
`4b02d15d6a68700181b180235be62a9fb95d2a38`이다.

## Manifest file hashes와 row-integrity audit

| Exp | `run_manifest.json` SHA-256 | `data_manifest.json` SHA-256 |
|---|---|---|
| E22a | `96dcde741e43d80ea7fe005d02c17896db1100c0ce58bd0fd75d8a4119dc8186` | `8145fdcab6a60d502fae15df8a7bc08efbbb159d6ddd2f386e057d80f71cb647` |
| E22b | `29389466d2cdcd5fd32179002e561bd2ae1a2814b32c9a8b7c8e3b74e7f02f14` | `5767c7887b99bcff0b1cb9bb8b8fe9a9cd452ad78c780bb171f9fefbfa755d0f` |
| E23a | `9ac417a380eb54a6dd468ae9da615d18fbe38430538ddbb4180951e552fbce03` | `a20c16949ac69bd4d5e0d6b1a49519df6b356156615a9a2f0dc46c8a921a1b87` |
| E23b | `14cdb2ae8feb20cb08de7eaea0eb75639e74a97dec88e0d2c653295aabc1cdb7` | `acd6e11a2462d579f487a47bef89f7b34a7e046bd9fe0066f86a9ff72409f8be` |
| E24a | `ab7f72a18e58b2554048a26316844b4b77d1d3223f63e3f8ade0fb927089ca40` | `02a212808e28585258fb5eb76991e295f9b5099af205ce6a9dd0f81ef8a3cfb5` |
| E24b | `8b411e8c0b2128d28ab8ff7b858cd553417ed0b790a82b5c1774d7ca91961342` | `191e3bd4b7bb627a09632b917178d86e3dc69d68cbf7d2b176e810c87e2013b1` |
| E25a | `357faec3b83b66a69433d679e3b4b9e1bc78156c5fbb6c0f68166b911f57e206` | `5376ef6b68a9e246f6790afc6c7e1f8a2b46944e9faf8b6cfca5834e5d6493f7` |
| E25b | `f3483840f787ac030635defa01ccaa3c4b0b0c9fed94876351aed2d96bde2d35` | `d1298b91dab16f9e665b17d77bdcaa03cea1e05242e8661fd400e2262f44c303` |

모든 8개 canonical run에서 manifest/report JSON parse, exact `latest.json`,
registered row count, duplicate 0, non-finite 0, missing row 0을 확인했다.
E22b와 E23b는 각각 독립 checkpoint/shard deep audit까지 수행했다.

## Canonical artifact 표

| Exp | Run | Report SHA-256 | Protocol SHA-256 | Rows | Checkpoint aggregate |
|---|---|---|---|---|---|
| E22a | `20260728T174341.363711Z` | `00ca1962a6829daceb74cf7fd4a54de544d782df57ae6cc463b0ee6f410b3ba9` | `1012bea7d097d8b0079273883384f28ad0f31a4a1f41a2164e481f5b9ebf0244` | raw 25,344; seed 33; active 11,088 | `3fa1378327b540ddce9b10e6d7643718cf0c4e626b9f8b7f01214d5388414f77` |
| E22b | `20260730T005512.389198Z` | `56f65f8519dc69a981c477c17bced73319a741108282444d756a9572209702f7` | `e19dfd26018e53d7ab601d1bd1b0e94c3bd922e1849c35cdcffec7ae38474598` | raw 12,288; seed 16; active 5,376 | `f940b2de69ef2d832bcd28c4ab0855124c177b0b433b9ce3d684efe16da1a97c` |
| E23a | `20260728T174341.359934Z` | `d0de1b91b3ec04a5b806357f72576e1de7399176a62e06326d64078e34325b7d` | `93f7a4e7c7df2505d984e887b8b0308f8d1e889faa45bce4c377530e83f8aa6e` | raw 14,256; seed 3 | `79f2b049a2c0950f333ec497905b2f377c9482f5df1e7418b8aeac81bfd9ac6c` |
| E23b | `20260730T113127.332498Z` | `5dea3af8d434a8963198f6b1a912c2bb8cc55357845342723a74a85c58d1d11d` | `ffcff888c026a4cbf03e6443965f49f42d3fbd9034d6f7374e1f5203fb02b1ca` | raw 38,016; training 128; seed 8; cells 88 | `42f54a7e09ad6c8954289a4d82f9b82087fdd48b6f243ce3182df35039b06d3a` |
| E24a | `20260728T174341.503129Z` | `3acf8be7877c27ec9aa7e23fe083467508bdaba2567c8ab7a24f4290f685a0ba` | `981d8c0383db519775508dc5585461691108f7e51cafe06c4dbf4dfa8627d52e` | raw 960; seed 8 | `ba53d4e170f5812abd590b9e256800d880be5a653e80668266116f4c3ca648a1` |
| E24b | `20260728T174341.918677Z` | `d4bd3f8bc21a4171f9c0ef9f4fa6e69f415236b74c3e6ac2995d02ef528f8abb` | `233caf6a55748766c800dab2c8c59f8bec80b2a54f243f3b97dd9e2a0f1e0fcb` | raw 69,120; seed 8 | `75f3d930f3d6e8bf5ff5f2448b970597e255736e81d883e0d34928b9599a8daf` |
| E25a | `20260728T130831.653416Z` | `3e071908cc849844312cb1223c64a6b7f5fbb3ae1eb831313d27a3f6d456bc41` | `f6c8c966a97266b680458589992f25bce4940a851375b7084f8922254bb36262` | raw 1; seed 1 | 없음 |
| E25b | `20260728T174341.332797Z` | `4fc0dcbe5c5d8b5693adffd96b203bd245abff8376e74989208b1a33ae1159d2` | `398882e4a8c3f18449c7d04698d432328eb758b0c6a10a2ec13c1679ab8f0610` | scientific raw 0; seed 0; audit 300 | 없음 |

## Raw/seed hash 표

| Exp | Raw SHA-256 | Seed/training SHA-256 | 1-page summary SHA-256 |
|---|---|---|---|
| E22a | `74b9bfe071c68d2bb529b3283887d2478a45b5666ee02549b0ec8f384ade7a31` | `2dbe563e8578423797d5f5a36287ed181196ca475fd4ba591688cc15b9097ab2` | `6d4e587931f8196caa5886771eccad0214d9e63db44408dc8546003386074b42` |
| E22b | `779a9806b8417fdba720cf26a534051a11f7fe9086de5ac835cc056ec245682f` | `ad47d347c403218c388c7684abd27679981528360ea0469b5f2c5304ccdb859b` | `bf20296e558003f084f16d411d273e9cae633fe55a85abe5c312f53bb78a76d2` |
| E23a | `1b3fb4d5d6db54854c66526ee7530a6624163347519649c88726ceed22f175cc` | `2e0290e205f9160251220a513b56521150fcb65622727bc1acc7fd295784874d` | `26024bacb49ca1d1d57ff9e958e403e61da2559bd2a8e72c5323789f3dd4f7e7` |
| E23b | `2b56fa2c03125f9e72d669efa506bc6eac451eb89664433475c394aa47951c82` | training `ab93e2205ff84913c273212863393b64a356417ead4569ede074d5c29ed7e669`; seed `3b6307a105410240afafefc478d7c2d7138031773a0e962b35fcff4ad3e66f3f` | `bd34bf03c17fcdffaaa310f4d56dba3b577429a716c2e24c1e50b0b4ddf287e4` |
| E24a | `a2f41e21c51b913b1d78575d78fc0181368ed5229e4ce0ef5084f70fe34367f1` | `a0ad0c66950bed7dfe3376d59157009c4777ab525b01c8c88f5812e9f5d6c9de` | `7d647ef606c5faf705f091fe79426f660491f4f1a09ebf945161401c5ec52a68` |
| E24b | `949e38b04f235ba2ece88cbe7d0d03a00fc64389a7c6e0f27555139dc89b9391` | `45f292e27f19c1ce58c748e994b1198c520010a0e2afb1fe66dc5bed77c9b36b` | `bd021dd74aaa03b2cbf2d7ac37f594e0ea5237e0eb481a75803e09fe40ed47f2` |
| E25a | `37a17e2ccb05815e5a5dc70bc96798d7ad6141a07c7f1bd463c3a71bf99c07db` | `c249a9e6171c838b85c375186c52cc9711c44afd23caf718a2617c079a9f1703` | `3ead62e2b23cf96d6afc6f64118245e598e936a8dc477edc18f2855c348830ec` |
| E25b | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `67ac7d1938035870ee99adc9bf5fe9ea9f19ba62cd0a31c751a957e8ec9670e2` |

E22b 추가 artifact:

- Active-cell 5,376 rows:
  `8bfd02d6bcce2eddca8ed2934a2601b30c127c91d77ada6ae40ccafe2bf25c3d`
- Execution amendment:
  `a6584ebd0ba91bdb73c7b6155f30944cdb10375b1f2ff172053999973138f6e2`
- CPU serial/shard equivalence:
  `945507ec26ce568ebef95764db59308c048a1c09685d7ef47535f9fe6f31e02e`
- Canonical checkpoint files: 64
- Exact run file count: 167
- Independent audit: 287/287 checks PASS. 이 검사는 additive validation이며
  frozen E22b run 내부에 별도 receipt를 사후 삽입하지 않았다.

E23b 추가 artifact:

- Poset cell 88 rows:
  `d8d17d58488ebbfdf937b7fa0a1e6e76776c2c3909bd52d1f37b81f2fde4a562`
- Execution amendment:
  `9f9c6512d9287bca4fa85501c2a6271134a3048a45520cedcfa2792815046209`
- CPU serial/shard equivalence:
  `15a3821b009d3c6be84471d487922d4b784630818169398f69fbc974b0dfd18a`
- Canonical checkpoint files: 128
- Shard manifests: 4
- Aggregate receipt report SHA binding:
  `5dea3af8d434a8963198f6b1a912c2bb8cc55357845342723a74a85c58d1d11d`
- Exact canonical run file count: 147
- Run file-map aggregate:
  `ee7e38eb74a324e3a9556e7714708904ed8dc110740665ae9d4baa93c7a0819c`
- Independently loaded checkpoint state-map aggregate:
  `e1bd96c88eb8d811a2a6c44a879ccaa278e7e4948ad7853b152068f1ca4d0aa0`
- Independent audit: raw/training/seed/cell Cartesian missing 0,
  duplicate 0, non-finite 0, checkpoint file/state 128/128 exact, latest pointer
  exact

## E25b human-audit package

Package:
`/data/minjun_dev/CATENA/artifacts/e25b_human_audit_packages/20260728T174341Z`

| File | SHA-256 |
|---|---|
| `PACKAGE_MANIFEST.json` | `2976a2073a0f3813a273b52811ff5ed0961e25bb71aaea12a1288d851341565d` |
| `REVIEWER_INSTRUCTIONS_KO.md` | `03e8c8271cb52dcf95efb5270e899e8c49aa9c960b4ed3d987196bbdd10b9adb` |
| `merge_validate_audit.py` | `3e20a85767e7d65bf058f407f100c9b461d5f9d95d11dc39b8ff22c602656449` |
| Original reviewer CSV | `4c84d81f547876d50b58087e5e8edb5093d43c77a7dad05b3adc05d2d224dae6` |
| Reviewer A CSV | `4c84d81f547876d50b58087e5e8edb5093d43c77a7dad05b3adc05d2d224dae6` |
| Reviewer B CSV | `4c84d81f547876d50b58087e5e8edb5093d43c77a7dad05b3adc05d2d224dae6` |

세 CSV는 300 data row와 header를 가지며 최초 byte content가 동일하다.
Reviewer label은 모두 비어 있다. Reviewer A/B는 실제 독립된 인간이어야
하며, merge 이후 disagreement에 한해 제3의 인간 adjudication을 사용한다.

Reviewer A/B exact paths:

```text
/data/minjun_dev/CATENA/artifacts/e25b_human_audit_packages/20260728T174341Z/reviewer_a/E25B_V4_HUMAN_AUDIT_REVIEW_WORK.csv
/data/minjun_dev/CATENA/artifacts/e25b_human_audit_packages/20260728T174341Z/reviewer_b/E25B_V4_HUMAN_AUDIT_REVIEW_WORK.csv
```

검수 열은 reviewer별 prefix를 갖는다.

```text
reviewer_{a,b}_semantic_preservation
reviewer_{a,b}_operation_leakage
reviewer_{a,b}_entity_ambiguity
reviewer_{a,b}_old_value_leakage
reviewer_{a,b}_gold_consistency
```

허용 label은 대문자 `PASS` 또는 `FAIL`뿐이다. 결측, `N/A`, 숫자와 임의
문자열은 금지된다. Merge/kappa/adjudication/validate 명령은
`/data/minjun_dev/CATENA/artifacts/e25b_human_audit_packages/20260728T174341Z/REVIEWER_INSTRUCTIONS_KO.md`
에 고정했다. Adjudication CSV는 두 reviewer 완료 후 merge 명령이
disagreement에 대해서만 생성한다.

## E00–E21 불변성

`docs/POST_E21_PREIMPLEMENTATION_ARTIFACT_SHA256.json`을 기준으로
`scripts/verify_pre_e22_artifacts.py`를 재실행했다.

| Check | 결과 |
|---|---|
| Expected / observed files | 1,329 / 1,329 |
| Missing / unexpected / changed | 0 / 0 / 0 |
| Expected aggregate | `4ef82dd6fe42543bc1e3b8a63f24781c2ba5d476a39bd84448f9dde6896c87f7` |
| Observed aggregate | `4ef82dd6fe42543bc1e3b8a63f24781c2ba5d476a39bd84448f9dde6896c87f7` |
| Status | `PASS` |

결과 파일:

- `docs/POST_E21_ARTIFACT_HASH_VERIFICATION.json`
- `docs/POST_E21_ARTIFACT_HASH_VERIFICATION_KO.md`

## 최종 audit 판정

- 모든 MAIN artifact는 새 UTC run directory에 생성됐다.
- E00–E21 artifact는 변경되지 않았다.
- E22b/E23b의 paired-seed sharding은 사전 amendment 및 exact CPU
  equivalence에 결속됐다.
- E22b/E23b canonical raw Cartesian grid, checkpoint, shard manifest와
  latest pointer가 완결됐다.
- E25a terminal FAIL은 재실행·완화되지 않았다.
- E25b leakage gate와 MAIN은 실행되지 않았다.
- Dry-run 또는 audit-preparation 수치는 scientific evidence로 승격되지
  않았다.

## Additive errata

Immutable artifact는 수정하지 않는다.

- E24a MAIN report의 `claim_boundary.allowed_claim`에
  `"None; dry-run is non-evidence."`가 남아 있다. `run_mode=MAIN`,
  negative disposition과 `claim_eligible=false`를 우선해 positive claim을
  열지 않는다.
- E23a `RESULTS_SUMMARY_KO.md`의 `E22b dependency` label은 E18b freeze
  validation의 오기다. Report/data dependency와 실제 DAG는 E18b로
  올바르게 결속돼 있다.
