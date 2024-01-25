#adult 
adult_race_grouping=  {
        1.0:1.0,
        2.0:2.0,
        3.0:3.0,
        4.0:3.0, # mapping "American Indian Alone" to other
        5.0:3.0, # mapping "Alaska Naive Alone" to other
        6.0:6.0, 
        7.0:3.0, # mapping "Native Hawaiian Alone" to other
        8.0:3.0, # mapping "Some Other Race alone" to other
        9.0:3.0, # mapping "Two or More Races" to other
    }

ACSIncome_categories = {
    "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: "American Indian or Alaska Native, not specified",
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}


        
ACSPublicCoverage_categories = {
    #disability
    "DIS": {1.0: "With a disability", 2.0: "No disability"},
    # Employment status of parents
    "ESP":{
        0.0: "(not own child of householder, and not child in subfamily)",
        1.0: "Living with two parents: Both parents in labor force",
        2.0: "Living with two parents: Father only in labor force",
        3.0: "Living with two parents: Mother only in labor force",
        4.0: "Living with two parents: Neither parent in labor force",
        5.0: "Living with father: In labor force",
        6.0: "Living with father: Not in labor force",
        7.0: "Living with mother: In labor force",
        8.0: "Living with mother: Not in labor force",
    },
    # citizenship status
    "CIT":{
        1.0: "Born in the United States",
        2.0: "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or Northern Marianas",
        3.0: "Born abroad of U.S. citizen parent or parents",
        4.0: "U.S. citizen by naturalization",
        5.0: "Not a U.S. citizen",
    },
    #Mobility status (lived here 1 year ago)
    "MIG":{
        1.0: "Yes, same house (nonmovers)",
        2.0: "No, outside US and Puerto Rico",
        3.0: "No, different house in US or Puerto Rico",
    },
    # Military service

    "MIL":{
        0.0: "N/A (less than 17 years old)",
        1.0: "Now on active duty",
        2.0: "On active duty in the past, but not now",
        3.0: "Only on active duty for training in Reserves/National Guard",
        4.0: "Never served in the military",
    },
    # Ancestry Recode: 
    "ANC":{
        1.0: "Single",
        2.0: "Multiple",
        3.0: "Unclassified",
        4.0: "Not reported",
    },
    # nativity
    "NATIVITY":{
        1.0: "Native",
        2.0: "Foreign born",
    },
    # Hearing Difficulty
    "DEAR":{
        1.0: "Yes",
        2.0: "No",
    },
    # Vision Difficulty
    "DEYE":{"Yes": 1.0, "No": 2.0},
    # Employmnet Status Recode
    "ESR":{0.0: "N/A (less than 16 years old)",
        1.0: "Civilian employed, at work",
        2.0: "Civilian employed, with a job but not at work",
        3.0: "Unemployed",
        4.0: "Armed forces, at work",
        5.0: "Armed forces, with a job but not at work",
        6.0: "Not in labor force",
        },
# Gave Birth to child within the past 12 months
    "FER":{
        0.0: "N/A (less than 15 years old)",
        1.0: "Yes",
        2.0: "No",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: "American Indian or Alaska Native, not specified",
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}

ACSEmployment_categories = {
        # 'RELP',
        # 'DIS',
        # 'ESP',
        # 'CIT',
        # 'MIG',
        # 'MIL',
        # 'ANC',
        # 'NATIVITY',
        # 'DEAR',
        # 'DEYE',
        # 'DREM',
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: "American Indian or Alaska Native, not specified",
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}