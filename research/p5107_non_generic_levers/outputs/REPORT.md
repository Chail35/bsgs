# p=5107 non-generic ECDLP levers — executable report

## Executive result
- CM scalar collapse: omega acts as lambda=2342 mod 4993; every a+b*omega action is one scalar.
- First point-side factor-base decomposition found: quotient_prefix_32 with 3 terms and 544 group additions; derived k=3955.
- Held-out verification only after construction: k match=True, quotient index=616, PhaseWords=(56,5).
- Kani: omega(G) is dependent, and no nontrivial 2-,3-,13-torsion is rational because #E(F_p)=4993 is prime.
- Cheon: best hypothetical auxiliary-input d is 78 with cost proxy 16.833, but fixed endomorphisms preserve affine degree in the secret.
- GLV lattice: max minimal coefficient bound B=40; MITM cost 2B+1=81=Theta(sqrt(N)).

## Factor-base measurements
- **quotient_reps_832**: |F|=832, x=832, t=832; 1-sum 832/4993=0.1666, 2-sum 4992/4993=0.9998, 3-sum 4993/4993=1.0000.
- **suborbit13_reps**: |F|=13, x=13, t=13; 1-sum 13/4993=0.0026, 2-sum 88/4993=0.0176, 3-sum 407/4993=0.0815, 4-sum 1376/4993=0.2756.
- **suborbit64_reps**: |F|=64, x=64, t=64; 1-sum 64/4993=0.0128, 2-sum 1728/4993=0.3461, 3-sum 4993/4993=1.0000.
- **suborbit13_mu6_saturated**: |F|=78, x=39, t=13; 1-sum 78/4993=0.0156, 2-sum 2419/4993=0.4845, 3-sum 4993/4993=1.0000.
- **suborbit64_mu6_saturated**: |F|=384, x=192, t=64; 1-sum 384/4993=0.0769, 2-sum 4993/4993=1.0000.
- **quotient_prefix_4**: |F|=4, x=4, t=4; 1-sum 4/4993=0.0008, 2-sum 10/4993=0.0020, 3-sum 20/4993=0.0040, 4-sum 35/4993=0.0070.
- **quotient_prefix_8**: |F|=8, x=8, t=8; 1-sum 8/4993=0.0016, 2-sum 36/4993=0.0072, 3-sum 119/4993=0.0238, 4-sum 320/4993=0.0641.
- **quotient_prefix_13**: |F|=13, x=13, t=13; 1-sum 13/4993=0.0026, 2-sum 91/4993=0.0182, 3-sum 443/4993=0.0887, 4-sum 1583/4993=0.3170.
- **quotient_prefix_16**: |F|=16, x=16, t=16; 1-sum 16/4993=0.0032, 2-sum 136/4993=0.0272, 3-sum 782/4993=0.1566, 4-sum 2873/4993=0.5754.
- **quotient_prefix_32**: |F|=32, x=32, t=32; 1-sum 32/4993=0.0064, 2-sum 508/4993=0.1017, 3-sum 3564/4993=0.7138, 4-sum 4993/4993=1.0000.
- **quotient_prefix_64**: |F|=64, x=64, t=64; 1-sum 64/4993=0.0128, 2-sum 1808/4993=0.3621, 3-sum 4991/4993=0.9996, 4-sum 4993/4993=1.0000.
- **t_height_le_4**: |F|=6, x=3, t=1; 1-sum 6/4993=0.0012, 2-sum 19/4993=0.0038, 3-sum 37/4993=0.0074, 4-sum 61/4993=0.0122.
- **t_height_le_8**: |F|=18, x=9, t=3; 1-sum 18/4993=0.0036, 2-sum 163/4993=0.0326, 3-sum 823/4993=0.1648, 4-sum 2503/4993=0.5013.
- **t_height_le_16**: |F|=42, x=21, t=7; 1-sum 42/4993=0.0084, 2-sum 817/4993=0.1636, 3-sum 4429/4993=0.8870, 4-sum 4993/4993=1.0000.
- **t_height_le_32**: |F|=66, x=33, t=11; 1-sum 66/4993=0.0132, 2-sum 1759/4993=0.3523, 3-sum 4987/4993=0.9988, 4-sum 4993/4993=1.0000.
- **t_height_le_64**: |F|=120, x=60, t=20; 1-sum 120/4993=0.0240, 2-sum 3907/4993=0.7825, 3-sum 4993/4993=1.0000.
- **t_height_le_128**: |F|=240, x=120, t=40; 1-sum 240/4993=0.0481, 2-sum 4993/4993=1.0000.
- **t_height_le_256**: |F|=480, x=240, t=80; 1-sum 480/4993=0.0961, 2-sum 4993/4993=1.0000.
- **x_height_le_4**: |F|=12, x=6, t=6; 1-sum 12/4993=0.0024, 2-sum 73/4993=0.0146, 3-sum 304/4993=0.0609, 4-sum 927/4993=0.1857.
- **x_height_le_8**: |F|=20, x=10, t=10; 1-sum 20/4993=0.0040, 2-sum 195/4993=0.0391, 3-sum 1148/4993=0.2299, 4-sum 3631/4993=0.7272.
- **x_height_le_16**: |F|=36, x=18, t=18; 1-sum 36/4993=0.0072, 2-sum 585/4993=0.1172, 3-sum 3676/4993=0.7362, 4-sum 4993/4993=1.0000.
- **x_height_le_32**: |F|=72, x=36, t=36; 1-sum 72/4993=0.0144, 2-sum 2035/4993=0.4076, 3-sum 4993/4993=1.0000.
- **x_height_le_64**: |F|=150, x=75, t=75; 1-sum 150/4993=0.0300, 2-sum 4525/4993=0.9063, 3-sum 4993/4993=1.0000.
- **x_height_le_128**: |F|=244, x=122, t=120; 1-sum 244/4993=0.0489, 2-sum 4983/4993=0.9980, 3-sum 4993/4993=1.0000.

## Semaev/Groebner receipts
```json
{
  "S3": {
    "degrees": [
      2,
      2,
      2
    ],
    "total_degree": 4,
    "terms": 9
  },
  "S4": {
    "degrees": [
      4,
      4,
      4,
      4
    ],
    "total_degree": 12,
    "terms": 191,
    "construction_seconds": 0.1588271179999765
  },
  "benchmarks": [
    {
      "factor_base": "t_height_le_4",
      "x_degree": 3,
      "S3_specialized_degrees": [
        2,
        2
      ],
      "bezout_monomial_bound": 16,
      "groebner_status": "completed",
      "groebner_seconds": 0.005082140000013169,
      "basis_count": 1,
      "basis_degrees": [
        0
      ],
      "basis_terms": [
        1
      ],
      "resultant_status": "completed",
      "resultant_seconds": 0.00481443399996806,
      "resultant_degree": 6,
      "resultant_terms": 7,
      "gcd_degree": 0
    },
    {
      "factor_base": "t_height_le_8",
      "x_degree": 9,
      "S3_specialized_degrees": [
        2,
        2
      ],
      "bezout_monomial_bound": 100,
      "groebner_status": "completed",
      "groebner_seconds": 0.007905259999972714,
      "basis_count": 1,
      "basis_degrees": [
        0
      ],
      "basis_terms": [
        1
      ],
      "resultant_status": "completed",
      "resultant_seconds": 0.01035739100001365,
      "resultant_degree": 18,
      "resultant_terms": 19,
      "gcd_degree": 0
    },
    {
      "factor_base": "x_height_le_16",
      "x_degree": 18,
      "S3_specialized_degrees": [
        2,
        2
      ],
      "bezout_monomial_bound": 361,
      "groebner_status": "completed",
      "groebner_seconds": 0.04800942000002806,
      "basis_count": 1,
      "basis_degrees": [
        0
      ],
      "basis_terms": [
        1
      ],
      "resultant_status": "completed",
      "resultant_seconds": 0.028661353999950734,
      "resultant_degree": 36,
      "resultant_terms": 37,
      "gcd_degree": 0
    },
    {
      "factor_base": "t_height_le_32",
      "x_degree": 33,
      "S3_specialized_degrees": [
        2,
        2
      ],
      "bezout_monomial_bound": 1156,
      "groebner_status": "completed",
      "groebner_seconds": 0.18440374200002907,
      "basis_count": 1,
      "basis_degrees": [
        0
      ],
      "basis_terms": [
        1
      ],
      "resultant_status": "completed",
      "resultant_seconds": 0.09899004499999364,
      "resultant_degree": 66,
      "resultant_terms": 67,
      "gcd_degree": 0
    },
    {
      "factor_base": "suborbit13_mu6_saturated",
      "x_degree": 39,
      "S3_specialized_degrees": [
        2,
        2
      ],
      "bezout_monomial_bound": 1600,
      "groebner_status": "completed",
      "groebner_seconds": 0.284276675000001,
      "basis_count": 2,
      "basis_degrees": [
        1,
        1
      ],
      "basis_terms": [
        2,
        2
      ],
      "resultant_status": "completed",
      "resultant_seconds": 0.1524948629999585,
      "resultant_degree": 78,
      "resultant_terms": 79,
      "gcd_degree": 1
    }
  ]
}
```

## Kani torsion inventory
```json
{
  "cm_roots": [
    {
      "zeta": 311,
      "lambda_mod_N": 2342,
      "lambda_poly": 0,
      "point": [
        622,
        530
      ]
    },
    {
      "zeta": 4795,
      "lambda_mod_N": 2650,
      "lambda_poly": 0,
      "point": [
        4483,
        530
      ]
    }
  ],
  "chosen_zeta": 311,
  "chosen_lambda": 2342,
  "rank_test_symbolic_determinant": "det([[1,lambda],[k,k*lambda]]) = 0",
  "random_rank_tests": [
    {
      "k": 2,
      "det": 0
    },
    {
      "k": 7,
      "det": 0
    },
    {
      "k": 123,
      "det": 0
    },
    {
      "k": 2026,
      "det": 0
    }
  ],
  "available_unknown_map_images": 1,
  "omega_image_is_redundant": true,
  "required_auxiliary_torsion_basis_rank": 2,
  "torsion_inventory": [
    {
      "m": 2,
      "gcd_m_group_order": 1,
      "frobenius_matrix": [
        [
          0,
          1
        ],
        [
          1,
          1
        ]
      ],
      "full_torsion_extension_degree": 3,
      "rational_m_torsion_points": 1
    },
    {
      "m": 4,
      "gcd_m_group_order": 1,
      "frobenius_matrix": [
        [
          0,
          1
        ],
        [
          1,
          3
        ]
      ],
      "full_torsion_extension_degree": 6,
      "rational_m_torsion_points": 1
    },
    {
      "m": 8,
      "gcd_m_group_order": 1,
      "frobenius_matrix": [
        [
          0,
          5
        ],
        [
          1,
          3
        ]
      ],
      "full_torsion_extension_degree": 12,
      "rational_m_torsion_points": 1
    },
    {
      "m": 16,
      "gcd_m_group_order": 1,
      "frobenius_matrix": [
        [
          0,
          13
        ],
        [
          1,
          3
        ]
      ],
      "full_torsion_extension_degree": 24,
      "rational_m_torsion_points": 1
    },
    {
      "m": 32,
      "gcd_m_group_order": 1,
      "frobenius_matrix": [
        [
          0,
          13
        ],
        [
          1,
          19
        ]
      ],
      "full_torsion_extension_degree": 48,
      "rational_m_torsion_points": 1
    },
    {
      "m": 64,
      "gcd_m_group_order": 1,
      "frobenius_matrix": [
        [
          0,
          13
        ],
        [
          1,
          51
        ]
      ],
      "full_torsion_extension_degree": 96,
      "rational_m_torsion_points": 1
    },
    {
      "m": 128,
      "gcd_m_group_order": 1,
      "frobenius_matrix": [
        [
          0,
          13
        ],
        [
          1,
          115
        ]
      ],
      "full_torsion_extension_degree": 192,
      "rational_m_torsion_points": 1
    },
    {
      "m": 3,
      "gcd_m_group_order": 1,
      "frobenius_matrix": [
        [
          0,
          2
        ],
        [
          1,
          1
        ]
      ],
      "full_torsion_extension_degree": 6,
      "rational_m_torsion_points": 1
    },
    {
      "m": 13,
      "gcd_m_group_order": 1,
      "frobenius_matrix": [
        [
          0,
          2
        ],
        [
          1,
          11
        ]
      ],
      "full_torsion_extension_degree": 12,
      "rational_m_torsion_points": 1
    }
  ],
  "structural_gap": "Kani graph kernels require the unknown map on a basis of smooth prime-power torsion; P=[k]G and omega(P) provide only one dependent N-torsion eigenline and no images on auxiliary torsion."
}
```

## Cheon prerequisite audit
```json
{
  "N_minus_1_factorization": [
    [
      2,
      7
    ],
    [
      3,
      1
    ],
    [
      13,
      1
    ]
  ],
  "best_d_if_auxiliary_existed": [
    {
      "d": 78,
      "estimated_given_auxiliary": 16.83256210825484
    },
    {
      "d": 64,
      "estimated_given_auxiliary": 16.832645413464757
    },
    {
      "d": 96,
      "estimated_given_auxiliary": 17.009783751773647
    },
    {
      "d": 52,
      "estimated_given_auxiliary": 17.01004283900152
    },
    {
      "d": 104,
      "estimated_given_auxiliary": 17.12693615332443
    },
    {
      "d": 48,
      "estimated_given_auxiliary": 17.12726364451629
    },
    {
      "d": 128,
      "estimated_given_auxiliary": 17.559331966661947
    },
    {
      "d": 39,
      "estimated_given_auxiliary": 17.559839624583056
    },
    {
      "d": 156,
      "estimated_given_auxiliary": 18.147416809889123
    },
    {
      "d": 32,
      "estimated_given_auxiliary": 18.14810118484675
    }
  ],
  "known_endomorphism_scalars": [
    1,
    114,
    2342,
    2343,
    2344,
    2650,
    2651,
    2652,
    4992
  ],
  "nonlinear_auxiliary_match_tests": [
    {
      "d": 2,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 1,
      "fraction": 0.00020032051282051281
    },
    {
      "d": 3,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 2,
      "fraction": 0.00040064102564102563
    },
    {
      "d": 4,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 3,
      "fraction": 0.0006009615384615385
    },
    {
      "d": 6,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 1,
      "fraction": 0.00020032051282051281
    },
    {
      "d": 8,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 1,
      "fraction": 0.00020032051282051281
    },
    {
      "d": 13,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 12,
      "fraction": 0.002403846153846154
    },
    {
      "d": 16,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 3,
      "fraction": 0.0006009615384615385
    },
    {
      "d": 32,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 1,
      "fraction": 0.00020032051282051281
    },
    {
      "d": 64,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 3,
      "fraction": 0.0006009615384615385
    },
    {
      "d": 128,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 1,
      "fraction": 0.00020032051282051281
    },
    {
      "d": 384,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 1,
      "fraction": 0.00020032051282051281
    },
    {
      "d": 416,
      "best_fixed_endomorphism_scalar": 1,
      "matching_nonzero_secrets": 1,
      "fraction": 0.00020032051282051281
    }
  ],
  "affine_closure_states_tested": 1002,
  "theorem": "Starting from G and P=[k]G, group addition and fixed CM endomorphisms produce only [a+b*k]G. They cannot produce [k^d]G for d>1 without a k-dependent operation."
}
```

## Novel GLV hidden-decomposition audit
```json
{
  "lambda": 2342,
  "kernel_lattice_basis_initial": [
    [
      4993,
      0
    ],
    [
      -2342,
      1
    ]
  ],
  "gauss_reduced_basis": [
    [
      49,
      -32
    ],
    [
      32,
      81
    ]
  ],
  "determinant": 4993,
  "max_minimal_Linf_bound": 40,
  "mean_minimal_Linf": 24.722611656318847,
  "representation_count_in_box": 6561,
  "group_order": 4993,
  "target_solution": {
    "a": 19,
    "b": 23,
    "derived_k": 3955,
    "lookups": 64,
    "table_size": 81,
    "verification": true,
    "held_out_k_match": true
  },
  "complexity": "A one-dimensional MITM table and one-dimensional scan each have 2B+1 = Theta(sqrt(N)) entries; CM gives a constant-factor GLV decomposition, not a sub-square-root algorithm."
}
```

## Strict receiver status
The experiments may derive the small-instance discrete logarithm by nonqualifying finite work, but they do not construct the normalized line/path tuple (mu1,w_R,PhaseWords) with the required uniform bound. The original strict receiver is therefore not claimed closed.
