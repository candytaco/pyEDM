import numpy
from typing import Union, List

def Embed(data: numpy.ndarray,
          columns: Union[int, List[int]],
          embeddingDimensions: int = 0,
          stepSize: int = -1,
          includeTime: bool = False) -> numpy.ndarray:
    '''Takens time-delay embedding on columns via numpy array operations.
       if includeTime True : insert data column 0 in first column
       nan will be present in |tau| * (E-1) rows.

       Returns:
           numpy.ndarray: Embedded data array of shape
                          (n_samples - abs(tau)*(E-1), n_cols*E)
                          or (n_samples - abs(tau)*(E-1), n_cols*E + 1) if includeTime=True
                          '''

    if embeddingDimensions < 1 :
        raise RuntimeError( 'Embed(): E must be positive.' )
    if stepSize == 0 :
        raise RuntimeError( 'Embed(): tau must be non-zero.' )

    selected_data = data[:, columns]
    n_rows, n_cols = selected_data.shape

    # Setup shift indices
    shiftVec = [i for i in range(0, int(embeddingDimensions * (-stepSize)), -stepSize)]

    # Create embedded array
    embedded_cols = []
    for col_idx in range(n_cols):
        for shift in shiftVec:
            shifted_col = numpy.full(n_rows, numpy.nan)
            if shift >= 0:
                if shift < n_rows:
                    shifted_col[shift:] = selected_data[:n_rows - shift, col_idx]
            else:
                if -shift < n_rows:
                    shifted_col[:shift] = selected_data[-shift:, col_idx]
            embedded_cols.append(shifted_col)

    result = numpy.column_stack(embedded_cols)

    if includeTime:
        # Prepend first column of original data
        result = numpy.column_stack([data[:, 0], result])

    return result
