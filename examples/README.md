# Examples

Run any of these from the repository root:

```bash
PYTHONPATH=src python3 examples/01_diagnose_a_submission.py
```

(or plain `python3 examples/...` once the package is installed with
`pip install -e .`)

| | |
|---|---|
| `01_diagnose_a_submission.py` | one submission, rendered report plus programmatic access |
| `02_grade_a_cohort.py` | the instructor's view: what is a whole class getting wrong, and where to teach next |
| `03_swap_the_backend.py` | ablation — the same session through the offline reasoner and through Claude |
| `04_scripted_session.py` | drive the interactive tutor from code, including an edit-and-retry cycle |
