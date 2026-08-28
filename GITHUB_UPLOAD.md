GitHub Upload Checklist

Before committing:

1. Confirm that no experiment outputs are staged:

   git status --short --ignored

2. Confirm ignored outputs include runs/, __pycache__/, and model tensor files.

3. Confirm the code still compiles:

   python -m compileall run_exp0.py refusal_repro scripts tests

Recommended initial commit:

  git add .gitignore README.txt EXPERIMENT0_STATUS.md GITHUB_UPLOAD.md requirements.txt run_exp0.py refusal_repro data scripts tests
  git commit -m "Initial Experiment 0 refusal direction scaffold"

Recommended branch name:

  git branch -M main

Add your GitHub remote:

  git remote add origin https://github.com/<USER>/<REPO>.git

Push:

  git push -u origin main

Do not commit:

- Hugging Face tokens
- .env files
- runs/
- downloaded model weights
- direction.pt or candidate_directions.pt
- other large generated artifacts
