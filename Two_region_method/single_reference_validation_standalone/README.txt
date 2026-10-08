Standalone two-region single-reference validation
=================================================

Files
-----
run_single_reference_validation.py
    Main script. This is the file to run.

two_region_validation_data.npz
    Raw RW P_st and MFPT arrays extracted from the low-, medium-, and
    high-barrier two-region branches. It does NOT contain the previously
    reconstructed two-reference beta U curves.

extract_data_from_bundle.py
    Optional provenance script. Use this if you want to recreate the NPZ
    directly from MFPT_all_branches.bundle.

Requirements
------------
Python 3 with:
    numpy
    scipy
    matplotlib

Direct run
----------
Keep run_single_reference_validation.py and two_region_validation_data.npz
in the same folder, then run:

    python run_single_reference_validation.py

The script prints the exact comparison table and creates:

    single_reference_validation_output/
        single_reference_boundary_results.csv
        boundary_values.png
        boundary_deviations.png
        matched_profile_low.png
        matched_profile_medium.png
        matched_profile_high.png

What is actually tested
-----------------------
1. RW free-energy profiles are reconstructed again from the raw saved P_st
   and MFPT arrays. The saved two-reference free-energy arrays are not used.
2. Region A is fixed using the single known value beta U(-1.0).
3. Six points nearest a=-0.1 are fitted with a degree-2 polynomial in x-a.
4. Region B is shifted so the two fitted one-sided values agree at a.
5. Only after this matching is complete is the analytical beta U(a) evaluated
   to quantify the validation deviation.
6. The TM benchmark is independently rebuilt inside the script.

Expected output
---------------
low:    RW deviation ~0.082221; TM deviation ~0.005235
medium: RW deviation ~0.015966; TM deviation ~0.005037
high:   RW deviation ~0.068273; TM deviation ~0.002627

Thus all three RW deviations are below 0.1 in beta U, while the TM deviations
are approximately 5e-3 or smaller.

Optional: recreate the raw-data NPZ from your Git bundle
--------------------------------------------------------
If MFPT_all_branches.bundle is available:

    python extract_data_from_bundle.py MFPT_all_branches.bundle

This overwrites/creates two_region_validation_data.npz in the current folder.
Then rerun the main script.
