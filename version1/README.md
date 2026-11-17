# Lower bounds for kissing numbers between dimensions 25 and 31 — version 1

Rustem Takhanov, Nazarbayev University.

This folder contains the paper (`paper.pdf`) and the exact certificates
(`certificates/`) for kissing configurations of the following sizes:

| d  | This release | Comparison bound | Gain  |
|----|-------------:|-----------------:|------:|
| 26 | 201 192      | 201 188          | +4    |
| 27 | 210 108      | 209 884          | +224  |
| 28 | 223 496      | 222 120          | +1376 |
| 29 | 232 153      | 231 814          | +339  |

The comparison bounds are those of
[alexlegeartis/KissingNumbers at commit a38c522](https://github.com/alexlegeartis/KissingNumbers/blob/a38c522d5a52e0e58f3988a10378dfaa35827bb1/RESULTS.md)
(accessed 16 November 2026), as in the paper; see `certificates/provenance/`.
A second, different 232 153-point witness in dimension 29 (`d29_alt`) is also included.

## Contents

- `README.md` — release contents and verification instructions.
- `paper.pdf` — the paper.
- `certificates/` — compact certificates, full coordinate files, verification
  scripts, and the reports of the recorded run. Start with `certificates/README.txt`.
- `SHA256SUMS.txt` — SHA-256 checksums of every other file in this folder,
  including `paper.pdf` and `certificates/SHA256SUMS.txt`.
- `.gitattributes` — stops git from converting line endings, so the checksums
  remain valid after cloning on any platform.
- `.gitignore` — keeps the outputs of a verification run (`__pycache__/`,
  `fresh_reports/`, …) out of the repository.

## Checking file integrity

Section 4 of `paper.pdf` prints the SHA-256 digest of `certificates/SHA256SUMS.txt`. From the `version1/` folder, run the commands in order. Compare the digest printed by the first command with the value in the paper. The second command uses that manifest to check the certificate files. The third checks the entire release, including the PDF.

On Linux or in Git Bash on Windows, run

```sh
sha256sum certificates/SHA256SUMS.txt
(cd certificates && sha256sum -c SHA256SUMS.txt)
sha256sum -c SHA256SUMS.txt
```

On macOS, run

```sh
shasum -a 256 certificates/SHA256SUMS.txt
(cd certificates && shasum -a 256 -c SHA256SUMS.txt)
shasum -a 256 -c SHA256SUMS.txt
```

## Checking the configurations

With Python ≥ 3.9 and NumPy ≥ 1.24:

```
cd certificates
python run_all.py --reports fresh_reports
```

This reconstructs every witness from its compact certificate, checks every
pair of points with exact arithmetic, checks the structural claims of the
paper, and runs rejection tests on invalid examples. It takes a few minutes
on one core and ends with `"status": "ALL CHECKS PASS"`. The individual steps
are described in `certificates/README.txt`.
