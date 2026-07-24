# Strict coordinate-factor-base index-calculus receipt

- Factor-base variables: 18 (36 signed points), rule `min(x,p-x)<=16`.
- Relation trials: 21; decomposed: 18; probability 0.857143.
- Independent relations: 18; all solved logs verified by point multiplication: True.
- Target decomposition nonzero terms: [(9, 1), (12, 4992), (17, 4992)].
- Derived scalar: `3955`; point verification: `True`.
- Point-BSGS baseline scalar: `3955`; baby entries `71`, giant steps `56`.
- Actual S4/Groebner benchmark: `{"S4_degrees": [4, 4, 4, 4], "S4_terms": 191, "S4_total_degree": 12, "boolean_box_monomial_bound": 6859, "factor_base_poly_degree": 18, "groebner_status": "timeout_120s"}`.

No target scalar, orbit index, PhaseWord, atlas, or precomputed factor-base logarithm was used by the construction.
The success is fixed-instance and non-generic, but not asymptotically faster: balanced m-sum MITM has exponent `ceil(m/2)/m >= 1/2`.
