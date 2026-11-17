EXACT CERTIFICATES -- 17 NOVEMBER 2026

Records: d26=201192, d27=210108, d28=223496, d29=232153.
Alternative witness: d29_alt=232153.

Start here
----------
Run from this directory (Python >=3.9 and NumPy >=1.24):
    OPENBLAS_NUM_THREADS=1 python run_all.py --reports fresh_reports

It reconstructs and verifies all five witnesses, including the supplied
dimension-29 tie, checks all structural claims, and runs invalid-example
rejection controls. Success ends with "status": "ALL CHECKS PASS". An error
raises an exception and the command exits nonzero. The delivered reports/
directory contains the completed checks for this version; fresh_reports/
is a separate output so that the recorded run is not overwritten.

Typical working memory is below 1 GB for the default block sizes. On the
assembly machine, full geometry alone took roughly 16--34 seconds per record
with one BLAS thread. Reconstruction additionally scans the shell against
the distinct added bodies. Runtime depends on the machine and NumPy build.
No GPU, solver, network, external repository, or original input archive is
needed after this package and NumPy are present.

Individual tasks
----------------
    python construct.py d26 --output rebuilt26.npz
    python verify.py d26 --report new_report26.json
    python check_structure.py d26 d27 d28 d29
    python negative_controls.py

Run only the four primary witnesses:
    python run_all.py --primary-only --reports fresh_primary

For a slow direct-int64 alternative to the exact binary64 integer kernel:
    python verify.py d29 --direct
    python construct.py d29 --direct --output rebuilt29.npz
    python run_all.py --direct --reports direct_reports

All scripts resolve their input data relative to their own location. Paths
passed as output/report arguments are relative to the caller's directory.

How to audit this package (also suitable for an LLM)
-------------------------------------------------
1. Read manifest.json and SCHEMA.txt. Select a record explicitly. d29 and
   d29_alt have equal sizes but are different coordinate sets.
2. Read common.py. It enumerates the extended Golay code and fixes the exact
   Leech-shell sign convention and numerical lexicographic indexing.
3. Read data/dN_certificate.npz using numpy.load(..., allow_pickle=False).
   Its added points and rational rows, together with the generated Leech
   shell and conflict-set rule, are the definition of the finite witness.
4. Run construct.py: it recomputes every deleted shell index, retains every
   compatible shell vector, reconstructs all coordinates, and compares the
   canonical hashes. It does not need the full coordinate file as input.
5. Run verify.py: it checks the distributed full coordinate file's identity,
   all norms, distinctness, and all unordered pairs. Because reconstruction
   produced the same canonical hashes, this checks the defined witness.
6. Read check_structure.py and its report. It checks the paper's layer table,
   midpoint endpoints, fibre occupancies, frames, cycles, cube exchange,
   and rational support/margin. These checks explain
   the construction; the complete geometry check does not rely on their
   labels or on assumptions about the search.

The successful reports are evidence of the included run, not substitutes for
re-executing the verification. A hash is an identity check, not a proof of a
kissing inequality. No approximate coordinate file is used for acceptance.

Exact arithmetic
----------------
All distributed vectors have squared norm 128 and require pair products <=64.
Every integer coordinate has absolute value <=8. For a binary64 dot product,
every product and every partial sum is therefore an exactly representable
integer, bounded in absolute value by 29*8*8=1856 < 2**53. The code checks its
applicable bound. There is no tolerance, rounding, or approximate threshold.
The --direct option replaces the large matrix kernel with checked int64.

Integer antipodal pairs are grouped only after distinctness is checked and
both signs have been found. All unpaired vectors retain their actual signs.
For P full pairs and U singles, the P/P absolute products, P/U absolute
products, and U/U signed products, plus the known products of the antipodes,
cover 4*C(P,2)+P+2*P*U+C(U,2)=C(2P+U,2) unordered pairs.

For a rational row n/q, the tests are n.n=128*q*q, n.v<=64*q,
and n_i.n_j<=64*q_i*q_j for i != j. Positive q is required. Norms and products between
rational rows use Python arbitrary-precision integers. Cross-products with
the integer core use int64 only after an explicit absolute-sum bound.

Only d26 has rational rows: 52 points supported on the eight octad coordinates
and the two tail coordinates. The original norm-32 input is multiplied by two
in the distributed norm-128 files. Denominators are unchanged by this scaling.
This normalization must not be confused with the integer files' inherited
coordinate_denominator=2 metadata, which is not a unit normalization.

Source and comparison provenance
--------------------------------
manifest.json records original filenames, their byte SHA-256 values, and
the coordinate scaling applied. The two supplied 29D witnesses share 228218
points, with 3935 exclusive to each. No isometry distinction is asserted.

provenance/comparison.json gives the fixed comparison revision and values.
A scoped excerpt of the source RESULTS.md, its source hash and pinned URL,
and the relevant revision metadata are included. They support attribution and the
comparison table; they are not dependencies of the geometric proof.

The package supplies certificates of the final finite sets, not claims that
the searches were exhaustive or that these lower bounds are optimal.
