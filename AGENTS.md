# Workspace Safety Rules

These rules apply to this repository and every directory below it.

## Filesystem scope

- Work only inside `/home/minjun_dev/`.
- Never read, create, modify, move, or delete files outside `/home/minjun_dev/`.
- Inside `/home/minjun_dev/`, files may be read, created, modified, moved, or deleted when needed for the user's request.
- In particular, repository inspection inside `/home/minjun_dev/CATENA/` is always allowed.

## Terminal commands

- Shell commands may be executed when their working directory, file inputs, and generated outputs are confined to `/home/minjun_dev/CATENA/`.
- Repository programs, scripts, tests, builds, and experiments may be executed when they operate only on this repository and its repository-local data, caches, logs, and artifacts.
- Read-only inspection of the existing Conda environment named `catena` and the host hardware/runtime required by repository experiments is allowed.
- Never run a command that changes global or system state, including system registries, system configuration, services, drivers, kernels, boot state, or files outside `/home/minjun_dev/`.
- Use dedicated file-editing tools rather than shell write tricks for source-controlled file edits.

## Python environment

- Use the existing Conda environment named `catena` for all Python commands.
- Prefer `conda run -n catena ...` so the selected environment is explicit and reproducible.
- Never create, activate, depend on, or write instructions for `.venv` or another virtualenv.
- The user has explicitly authorized installing, upgrading, downgrading, and
  removing packages inside the existing Conda environment named `catena`.
- Do not create another Conda environment or modify Conda `base` or any other
  environment.

## Privilege and system safety

- Never use `sudo`, `su`, privilege escalation, or elevated execution.
- Never reboot, shut down, suspend, or otherwise restart the machine.
- Never invoke system-wide service or power-management operations.

## Package and dependency safety

- Package operations are authorized only inside the existing Conda environment
  `catena`; no additional approval is needed for work required by this repository.
- Never install, upgrade, downgrade, or remove system packages or packages in
  Conda `base`, another Conda environment, or a global user environment.
- This includes system package managers and project/language package managers such as `apt`, `dpkg`, `snap`, `pip`, `pipx`, `conda`, `npm`, `pnpm`, `yarn`, `bun`, `cargo`, and similar tools.
- Dependency declarations required by this repository may be added or changed;
  any corresponding package operation must remain inside `catena`.

## Priority

- Treat these restrictions as hard safety boundaries for all work in this repository.
- If a requested action conflicts with these rules, stop and explain the conflict instead of performing the action.
