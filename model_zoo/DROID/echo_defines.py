category_dictionaries = {
    'view': {
        'PLAX': 0,
        'Ascending_aorta': 1,
        'RV_inflow': 2,
        'RV_focused': 3,
        'Pulmonary_artery': 4,
        'PSAX_AV': 5,
        'PSAX_MV': 6,
        'PSAX_papillary': 7,
        'PSAX_apex': 8,
        'A4C': 9,
        'A5C': 10,
        'A3C': 11,
        'A2C': 12,
        'Suprasternal': 13,
        'Subcostal': 14
    },
    'doppler': {
        'standard': 0,
        'doppler': 1,
        '3-D': 2
    },

    'quality': {
        'good': 0,
        'unusable': 1,
    },
    'canonical': {
        'on_axis': 0,
        'off_axis': 1
    },
    'LV_EjectionFraction': {
        'N': {
            'index': 0,
            'weight': 0.259667,
        },
        'A': {
            'index': 1,
            'weight': 0.862008,
        },
        'I': {
            'index': 2,
            'weight': 0.916131,
        },
        'L': {
            'index': 3,
            'weight': 0.980843,
        },
        'H': {
            'index': 0,
            'weight': 0.981351,
        }
    },
    'LV_FunctionDescription': {
        '4.0': {
            'index': 0,
            'weight': 0.520803,
        },
        '2.0': {
            'index': 1,
            'weight': 0.662169,
        },
        '3.0': {
            'index': 2,
            'weight': 0.817028,
        }
    },
    'LV_CavitySize': {
        'N': {
            'index': 0,
            'weight': 0.209487,
        },
        'D': {
            'index': 1,
            'weight': 0.833406,
        },
        'S': {
            'index': 2,
            'weight': 0.957354,
        },
        'P': {
            'index': 3,
            'weight': 1.0
        }
    },
    'RV_SystolicFunction': {
        'N': {
            'index': 0,
            'weight': 0.19156206811684748,
        },
        'Y': {
            'index': 1,
            'weight': 2.5944871794871798,
        },
        'A': {
            'index': 2,
            'weight': 4.161422989923915,
        },
        'L': {
            'index': 3,
            'weight': 8.256629946960423
        }
    },
    'hf_task': {
        'PvN': [0, 1],
        'IvN': [0, 2],
        'IPvN': [0, 1, 2],
        'I10y': [0, 2],
        'survival': [0, 1, 2, 3]
    },
    'hf_diag_type': {
        'hf_nlp': 'hf_nlp_cls',
        'primary': 'hf_primary_cls',
        'both': 'hf_both_cls'
    }
}
