tracks = {}

genelec = Req([
    'CS108', 'CS124', 'CS131', 'CS140', 'CS142', 'CS143', 'CS144', 'CS145', 'CS147', 'CS148', 'CS149', 'CS154', 'CS155',
    Req(['CS157', 'PHIL151'], 2),
    'CS164', 'CS166', 'CS167', 'CS168', 'CS190', 'CS205A', 'CS205B', 'CS210A', 'CS221',
    'CS223A', 'CS224N', 'CS224S', 'CS224U', 'CS224W', 'CS225A', 'CS227B', 'CS228', 'CS229', 'CS229T',
    'CS231A', 'CS231B', 'CS231M', 'CS231N', 'CS232', 'CS233', 'CS240', 'CS240H', 'CS242', 'CS243', 'CS244',
    'CS244B', 'CS245', 'CS246', 'CS247', 'CS248', 'CS249A', 'CS251', 'CS254', 'CS255', 'CS261', 'CS262',
    'CS263', 'CS264', 'CS265', 'CS266', 'CS267', 'CS270', 'CS272', 'CS273A', 'CS273B', 'CS274', 'CS276', 'CS279',
    'CS348B', 'CS348C',
    'CME108',
    'EE180', 'EE282', 'EE364A'
], 1)


'''
Graphics
'''
graphics_reqB = Req(['CS205A', 'CME104', 'CME108', 'MATH52', 'MATH113'], 1)
graphics_reqC = Req([Req(['CS131', 'CS231A'], 1),
           'CS233', 'CS268', 'CS348A', 'CS348B', 'CS348C', 'CS448'], 2)
tracks['graphics'] = Req([
    'CS148', 'CS248',                  # Track Requirement A
    reqB,                              # Track Requirement B
    reqC,                              # Track Requirement C
    Subreq([reqB, reqC, genelec],
           ['ARTSTUDI160', 'ARTSTUDI160170', 'ARTSTUDI160179',
            'CME302', 'CME306',
            'EE168', 'EE262', 'EE264', 'EE278', 'EE368',
            'ME101',
            'PSYCH30', 'PSYCH221'], 2) # Elective
])


'''
Computational Engineering
'''
tracks['compeng'] = Req([
    'EE108', # A
    'EE180', # A
    Req(['EE101A', 'EE101B', 'EE102A', 'EE102B'], 2), # B
    Req([
        # Digital Systems
        Req([
            Req(['CS140', 'CS143'], 1),
            'EE109',
            'EE271',
            Req([
                Req(['CS140', 'CS143'], 1),
                'CS144', 'CS149', 'CS240E', 'CS244', 'EE273', 'EE282'
            ], 2)
        ]),
        # Robotics
        Req([
            'CS205A',
            'CS223A',
            'ME210', 
            'ENGR105',
            Req(['CS225A', 'CS231A', 'ENGR205', 'ENGR207B'], 1)
        ]),
        # Networking
        Req([
            'CS140',
            'CS144',
            Req(['CS240', 'CS240E', 'CS241', 'CS244', 'CS244B', 'CS244E', 'CS249A', 'EE179'], 3)
        ])
    ], 1) # C
])
