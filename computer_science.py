from req_builder import Req, Subreq

TRACKS = {}
TRACKNAMES = [
    'graphics',
    'compeng',
    'biocomp',
    'hci',
    'info',
    'systems',
    'theory',
    'ai',
    'unspecialized'
]
TRACK_PRIORITIES = {
    'graphics': 2,
    'compeng': 1,
    'biocomp': 3,
    'hci': 4,
    'info': 7,
    'systems': 8,
    'theory': 5,
    'ai': 6,
    'unspecialized': 9
}

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
TRACKS['graphics'] = Req([
    'CS148', 'CS248',                  # Track Requirement A
    graphics_reqB,                              # Track Requirement B
    graphics_reqC,                              # Track Requirement C
    Subreq([graphics_reqB, graphics_reqC, genelec],
           ['ARTSTUDI160', 'ARTSTUDI160170', 'ARTSTUDI160179',
            'CME302', 'CME306',
            'EE168', 'EE262', 'EE264', 'EE278', 'EE368',
            'ME101',
            'PSYCH30', 'PSYCH221'], 2) # Elective
])


'''
Computational Engineering
'''
TRACKS['compeng'] = Req([
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


'''
Biocomputation
'''
biocomp_reqA = Req(['CS221', 'CS228', 'CS229', 'CS231A'], 1) # A
biocomp_reqB = Req(['CS262', 'CS270', 'CS273A', 'CS274', 'CS275', 'CS279'], 1) # B
biocomp_elec3 = Req([
    'CS108', 'CS124', 'CS131', 'CS140', 'CS142', 'CS143', 'CS144', 'CS145', 'CS147', 'CS148', 'CS149',
    'CS154', 'CS155', Req(['CS157', 'PHIL151'], 1), 'CS164', 'CS166', 'CS167', 'CS168', 'CS190',
    'CS205A', 'CS205B', 'CS210A', 'CS221', 'CS223A', 'CS224N', 'CS224S', 'CS224U', 'CS224W', 'CS225A',
    'CS227B', 'CS228', 'CS229', 'CS229T', 'CS231A', 'CS231B', 'CS231M', 'CS231N', 'CS232', 'CS233', 'CS240',
    'CS240H', 'CS242', 'CS243', 'CS244', 'CS244B', 'CS245', 'CS246', 'CS247', 'CS248', 'CS249A', 'CS251',
    'CS254', 'CS255', 'CS261', 'CS262', 'CS263', 'CS264', 'CS265', 'CS266', 'CS267', 'CS270', 'CS272', 'CS273A',
    'CS273B', 'CS274', 'CS275', 'CS276', 'CS279', 'CS348B', 'CS348C', 'CS371', 'CS374',
    'CME108',
    'EE180', 'EE263', 'EE282', 'EE364A',
    'BIOE101',
    'MS&E152', 'MS&E252', 'STATS206', 'STATS315A', 'STATS315B',
    'BIOMEDIN231', 'BIOMEDIN260',
    'GENE211'
], 1)
biocomp_elec4 = Req([
    'CS145', 'CS147', 'CS221', 'CS228', 'CS229', 'CS262', 'CS270', 'CS273A', 'CS273B', 'CS274', 'CS275',
    'CS279', 'CS371', 'CS374',
    'EE263', 'EE364A',
    'MS&E152', 'MS&E252',
    'STATS206', 'STATS315A', 'STATS315B',
    'BIOMEDIN231', 'BIOMEDIN260',
    'GENE211'
], 1)
biocomp_elec5 = Subreq([biocomp_elec4], [
    'BIOE222A', 'BIOE222B',
    'CHEMENG150', 'CHEMENG174',
    'APPPHYS294',
    'BIO104', 'BIO118', 'BIO129A', 'BIO129B', 'BIO188', 'BIO189', 'BIO214', 'BIO217', 'BIO230',
    'CHEM135', 'CHEM171',
    'BIOC218', 'BIOC241',
    'SBIO228'
], 1)
biocomp_elec6 = Req([
    'BIOE220', 'BIOE222A', 'BIOE222B',
    'CHEMENG150', 'CHEMENG174',
    'CS262', 'CS274', 'CS279', 'CS371', 'CS374',
    'ME281',
    'APPPHYS294',
    'BIO104', 'BIO112', 'BIO118', 'BIO129A', 'BIO129B', 'BIO158', 'BIO183', 'BIO188', 'BIO189', 'BIO214', 'BIO217', 'BIO230',
    'CHEM135', 'CHEM171',
    'BIOC218', 'BIOC241',
    'DBIO210',
    'GENE211',
    'SBIO228',
    'SURG101'
], 1)
TRACKS['biocomp'] = Req([
    Req([
        Req(['CHEM31A', 'CHEM31B']),
        'CHEM31X'
    ], 1),
    'CHEM33',
    Req([
        Req(['BIO41', 'BIO42']),
        Req(['HUMBIO2A', 'HUMBIO3A', 'HUMBIO4A'])
    ], 1),
    biocomp_reqA,
    biocomp_reqB,
    Subreq([biocomp_reqA, biocomp_reqB], ['CS124', 'CS145', 'CS147', 'CS148', 'CS248'], 1), # C
    biocomp_elec3, biocomp_elec4, biocomp_elec5, biocomp_elec6
])


'''
Human-Computer Interaction
'''
hci_reqB = Req(['CS377' + c for c in 'AEWUTVDPMCIBSF'] + [
    'CS142', 'CS148', 'CS194H', 'CS210A', 'CS376', 'CS448B', 'ME216M'
], 3)
TRACKS['hci'] = Req([
    'CS147',
    'CS247',
    hci_reqB
    # skipping track req C because specifications too vague
])


'''
Information
'''
info_reqB = Req([
        Req(['CS224N', 'CS224S', 'CS229', 'CS233'], 1),
        Req(['CS140', 'CS142', 'CS245', 'CS246', 'CS341', 'CS345', 'CS346', 'CS347'], 1),
        Req(['CS262', 'CS270', 'CS274'], 1),
        Req(['CS224W', 'CS276'], 1)
    ], 2)
TRACKS['info'] = Req([
    'CS124', # A
    'CS145', # A
    info_reqB,
    Subreq([info_reqB, genelec], [], 3)
])


'''
Systems
'''
sys_reqB = Req(['CS143', 'EE180'], 1)
sys_reqC = Subreq([sys_reqB], [
    'CS144', 'CS145', 'CS149', 'CS155', 'CS240', 'CS242', 'CS243', 'CS244', 'CS245', 'EE271', 'EE282'
], 2)
TRACKS['systems'] = Req([
    'CS140', # A
    sys_reqB,
    sys_reqC,
    Subreq([sys_reqC, genelec], [
        'CS240E', 'CS241', 'CS244E', 'CS316', 'CS341', 'CS343', 'CS344', 'CS345', 'CS346', 'CS347', 'CS349', 'CS448',
        'EE108', 'EE382C', 'EE384A', 'EE384B', 'EE384C', 'EE384S', 'EE384X'
    ], 3)
])


'''
Theory
'''
theory_reqB = Req(['CS167', 'CS168', 'CS255', 'CS261', 'CS264', 'CS265', 'CS268'], 1)
theory_reqC = Subreq([theory_reqB], [
    'CS143', 'CS155', Req(['CS157', 'PHIL151'], 1), 'CS166', 'CS205A', 'CS228', 'CS233', 'CS242', 'CS250',
    'CS251', 'CS254', 'CS259', 'CS262', 'CS263', 'CS266', 'CS267', 'CS354', 'CS355', 'CS357', 'CS358',
    'CS359', 'CS364A', 'CS367', 'CS369', 'CS374', 'MS&E310'
], 2)
TRACKS['theory'] = Req([
    'CS154', # A
    theory_reqB,
    theory_reqC,
    Subreq([theory_reqC, genelec], ['CME302', 'CME305', 'PHIL152'], 3)
])


'''
Artificial Intelligence
'''
ai_reqB = Req([
    'CS223A',
    'CS224M',
    'CS224N',
    'CS226',
    'CS227',
    'CS228',
    'CS229',
    Req(['CS131', 'CS231A'], 1)
], 2)
ai_reqC = Subreq([ai_reqB], [
    'CS124', 'CS205A', 'CS224S', 'CS224U', 'CS224W', 'CS225A', 'CS227B', 'CS231A', 'CS231B', 'CS231M', 'CS231N', 'CS262',
    'CS276', 'CS277', 'CS279', 'CS321', 'CS326A', 'CS327A', 'CS329', 'CS331', 'CS331A', 'CS371', 'CS374', 'CS379',
    'EE263', 'EE376A',
    'ENGR205', 'ENGR209A',
    'MS&E251', 'MS&E351',
    'STATS315A', 'STATS315B'
], 1)
TRACKS['ai'] = Req([
    'CS221',
    ai_reqB,
    ai_reqC,
    Subreq([ai_reqC, genelec], [
        'CS238', 'CS275', 'CS278', Req(['CS334A', 'EE364A'], 1),
        'EE278', 'EE364B',
        'ECON286',
        'MS&E252', 'MS&E352', 'MS&E355',
        'PHIL152',
        'PSYCH202', 'PSYCH204A', 'PSYCH204B',
        'STATS200', 'STATS202', 'STATS205'
    ], 3)
])


'''
Unspecialized
'''
unspecialized_reqB = Req(['CS140', 'CS143'], 1)
TRACKS['unspecialized'] = Req([
    'CS154',
    unspecialized_reqB,
    Subreq([unspecialized_reqB], ['CS144', 'CS155', 'CS242', 'CS244', 'EE180'], 1), # C
    Req(['CS221', 'CS223A', 'CS228', 'CS229', 'CS231A'], 1), # D
    Req(['CS145', 'CS147', 'CS148', 'CS248', 'CS262'], 1), # E
    genelec,
    genelec
])
