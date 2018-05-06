'''
Sample:
map = {
    'L1':{
        'op_list': ['conv', 'pool', 'activation'],
        'conv':{
            'stride': 2,
            'pad': 1,
            'val': 0,
        },
        'pool':{
            'stride': 2,
            'mode': 'max',
        },
        'activation': 'relu',

        'cache':{
            'W':, None,
            'b': None,
            'Z': None,
            'A': None,
            'dW': None,
            'db': None,
        },
    },
}
'''

test_initialize_map = {
    'L0':{
        'cache':{
            'A': None,
        },
    },

    'L1':{
        'op_list': ['conv', 'activation', 'pool'],
        'conv':{
            'stride': 2,
            'n_fh': 2,
            'n_fw': 2,
            'n_c': 2,
            'pad': 0,
            'val': 0,
        },
        'pool':{
            'stride': 1,
            'n_fh': 2,
            'n_fw': 2,
            'mode': 'max',
        },
        'activation': 'relu',

        'cache':{
        },
    },

    'L2':{
        'op_list': ['conv', 'pool', 'activation'],
        'conv':{
            'stride': 2,
            'n_fh': 2,
            'n_fw': 2,
            'n_c': 2,
            'pad': 0,
            'val': 0,
        },
        'pool':{
            'stride': 1,
            'n_fh': 2,
            'n_fw': 2,
            'mode': 'max',
        },
        'activation': 'relu',

        'cache':{
        },
    },
}
