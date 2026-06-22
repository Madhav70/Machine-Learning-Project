# Milestone Feedback

Project: P17 - ASI manufacturing COVID shock / recovery
Repo: `Madhav70/Machine-Learning-Project`
Milestone score locked: 11/20
Raw score before policy caps: 11/20
Band: `major_revision`
Reviewed at: 2026-05-09T01:58:41

This is the locked milestone evaluation for the May 6 milestone. The score is based on the latest repository snapshot available to the instructor review workflow when this feedback was generated.

## Graduating-Student Timeline

This team includes graduating student(s): Vikas Chaurasiya (ma2024).
To help us meet the May 15 grade-publishing deadline from OAA, please aim to submit the final version by May 13 if possible, and no later than May 14, 2026 at 11:59 PM IST.

## Rubric Breakdown

- Charter lock: 1/4. the charter is not recorded as approved; charter file exists and is not obviously template; the milestone manifest does not confirm charter lock
- Source access proof: 4/4. source/probe evidence is present and usable
- Baseline before sophistication: 2/4. `outputs/baseline_metric.json` is readable, but still has template/placeholder residue
- Reproducible dry run: 0/4. `uv run main.py` fails from a fresh copy of the repo
- Metric schema readiness: 4/4. `outputs/primary_metric.json` is readable and machine-checkable

## What To Fix Next

- Make the charter unambiguous: final question, dataset, primary metric, baseline, scope limits, and team roles should all be visible in `CHARTER.md`.
- Keep a simple baseline first. `outputs/baseline_metric.json` should contain a real metric name and value, not template text.
- Make `uv run main.py` work from a fresh clone. If the full data is large, the script should still run on a committed sample or a clearly reproducible download path.

## Reproducibility Error Observed

The reviewer ran `uv run main.py` from a fresh copy of the repo. The relevant tail of the error was:

```text
Using CPython 3.14.4
Creating virtual environment at: .venv
Downloading numpy (5.0MiB)
Downloading pandas (9.5MiB)
 Downloaded numpy
 Downloaded pandas
Installed 9 packages in 37ms
Traceback (most recent call last):
  File "/private/var/folders/t6/gytrx5s95txg4g8vkt7rnb980000gn/T/eco6810-milestone-P17-il_c7p1w/repo/main.py", line 7, in <module>
    import matplotlib.pyplot as plt # Added import
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'matplotlib'
```

## Final Phase Guidance

- First priority: make the project reproducible from a fresh clone with `uv run main.py`. Do this before adding more modeling complexity.
- This needs urgent repair. A simple, reproducible, well-explained project will score better than an ambitious project that cannot be run or verified.
- For the final submission, keep the repo as the source of truth: `README.md`, `CHARTER.md`, `main.py`, `outputs/`, `report.md`, and `AI_USAGE_LOG.md` should tell one consistent story.

Please treat this feedback as a way to make the final week calmer, not as a ceiling on the final project. A clear, reproducible, honestly interpreted final submission can still be strong.
