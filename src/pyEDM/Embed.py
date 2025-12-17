from pyEDM.AuxFunc import IsIterable


def Embed( dataFrame     = None,
           E             = 0,
           tau           = -1,
           columns       = "",
           includeTime   = False,
           pathIn        = "./",
           dataFile      = None ):
           # deletePartial = False ):
    '''Takens time-delay embedding on columns via pandas DataFrame.shift()
       if includeTime True : insert dataFrame column 0 in first column
       nan will be present in |tau| * (E-1) rows.'''

    if E < 1 :
        raise RuntimeError( 'Embed(): E must be positive.' )
    if tau == 0 :
        raise RuntimeError( 'Embed(): tau must be non-zero.' )
    if not columns :
        raise RuntimeError( 'Embed(): columns required.' )
    if dataFile is not None:
        dataFrame = read_csv( pathIn + dataFile )
    if not isinstance( dataFrame, DataFrame ) :
        raise RuntimeError('Embed(): dataFrame is not a Pandas DataFrame.')

    if not IsIterable( columns ) :
        columns = columns.split() # Convert string to []

    for column in columns :
        if column not in dataFrame.columns :
            raise RuntimeError(f'Embed(): {column} not in dataFrame.')

    # Setup period shift vector for DataFrame.shift()
    # Note that DataFrame.shift() indices are opposite the tau convention
    shiftVec = [ i for i in range( 0, int( E * (-tau) ), -tau ) ]

    df = dataFrame[ columns ].shift( periods = shiftVec ).copy()

    # Replace shifted column names x with x(t-0), x(t-1)...
    # DataFrame.shift() appends _0, _1... or _0, _-1 ... to column names
    # Use rsplit to split the DataFrame.shift() names on the last _
    colNamePairs = [ s.rsplit( '_', 1 ) for s in df.columns ]
    if tau > 0 :
        newColNames = [ ''.join( [s[0],'(t+',s[1].replace('-',''),')'] ) \
                        for s in colNamePairs ]
    else :
        newColNames = [ ''.join( [s[0],'(t-',s[1],')'] ) for s in colNamePairs ]

    df.columns = newColNames

    if includeTime :
        # First column of time/index merged into df
        df = dataFrame.iloc[ :, [0] ].join( df )

    return df
