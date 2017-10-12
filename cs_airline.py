#-------------------------------------------------------------------------------

print ('\n')

print ('STATUS:: Begin importing keras\n')
import keras
print ('STATUS:: End importing keras\n')

import math
import matplotlib.pyplot as plt
import numbers
import numpy
import os
import os.path
import pandas
import pandas
from sklearn.preprocessing import MinMaxScaler
import sys
import timeit

print ('\n')

#-------------------------------------------------------------------------------

# RETURN VALUE
# # seconds since 1 January 1970 00:00 hours (UTC)minus the number of leap
# seconds that have taken place since then.

def get_epoch_time_in_seconds ():
    return timeit.default_timer ()

#-------------------------------------------------------------------------------

# RETURN_VALUE
# Let fn be "./hoho.csv".  The return value is True if ./hoho.csv exists as a
# file with read permission

def file_is_readable (fn):
    
    return os.path.isfile (fn) and os.access (fn, os.R_OK)

#-------------------------------------------------------------------------------

# RETURN VALUE
# On success, return value is a valid data frame of type
# pandas.core.frame.DataFrame
# On failure, return value is None

def load_csv_file (csv_fn):

    print ('STATUS:: Begin loading CSV file\n')
    t0 = get_epoch_time_in_seconds ()
    
    # File must be readable

    if not file_is_readable (csv_fn):
        return None

    # Read into a Pandas data frame

    try:
        D_orig = pandas.read_csv (csv_fn)
    except:
        print ('ERROR:: Unable to read %s' % csv_fn)
        return None

    # Print a message
    
    t1 = get_epoch_time_in_seconds ()
    elapsed_time = t1 - t0
    
    print ('STATUS:: Loaded %s rows in %.3f s\n' % (len (D_orig), elapsed_time))
    
    # Are some values missing?

    num_missing = numpy.count_nonzero (D_orig.isnull ().values)

    if num_missing > 0:
        print ('ERROR:: Number of missing values : %s' % num_missing)
        return None

    # Change from integer to float
    
    D_orig['Passengers'] = D_orig['Passengers'].astype('float32')

    # Five-number summary

    P = D_orig.iloc[ :, 1 : 2]
    print ('STATUS:: Describing P(t) ...\n', P.describe (), '\n')
 
    print ('STATUS:: End loading CSV file\n')
    
    return D_orig
        
#-------------------------------------------------------------------------------

# RETURN VALUE
# -1 => trouble
# Non-negative value => Mean Absolute Scaled Error in the sense of Hyndman &
# Koehler

def compute_mase (P, P_f, start_of_training_set, forecast_horizon):

    # Sanity check

    m = start_of_training_set
    n = m + len (P_f)
    h = forecast_horizon
    
    if m <= 0:
        print ('ERROR:: Argument start_of_training_set incorrect')
        return -1

    if  n <= m:
        print ('ERROR:: Argument P_f has an incorrect length')
        return -1

    if h >= m:
        print ('ERROR:: Arguments start_of_training_set and forecast_horizon ',
               'not compatible')
        return -1

    if h <= 0:
        print ('ERROR:: Argument forecast_horizon <= 0')
        return -1
    
    # Example: Let start_of_training_set be 132. And let P_f have a length of 12
    # It means that P[0:132] is the training set, and P[132:132+12] is the
    # testing set

    # The numerator is the L1 norm of the array ( P_f - P[m:n] ) / (n-m)

    diff = numpy.subtract (P_f, P[m:n])
    numerator = numpy.linalg.norm (diff, 1) / (n - m)
    
    # Example: Let start_of_training_set be 132. Let the forecast horizon
    # be 36.  The denominator is L1 norm of the array (P[36:132] - P[0:132-36])

    diff = numpy.subtract (P[h:m], P[0:m-h])
    denominator = numpy.linalg.norm (diff, 1) / (m - h)

    if numpy.isclose(denominator, 0.0):
        print ('ERROR:: P is a constant function of t on the training set---',
               'out of the scope of MASE theory')
        return -1

    # Compute MASE

    mase = numerator / denominator

    return mase
        
#-------------------------------------------------------------------------------

# RETURN VALUE
# True => OK, False => trouble

def my_main ():
    
    print ('STATUS:: Begin of my_main\n')

    # Set some parameters

    FORECAST_HORIZON = 36
    NUM_HIDDEN_NEURONS = 512
    NUM_TIME_POINTS_PER_SUB_VECTOR = 3
    ACTIVATION = 'relu'
    
    # Load CSV file
    
    csv_fn = './airline.csv'
    D_orig = load_csv_file (csv_fn)

    if D_orig is None:
        return False

    # Get 2nd column (passengers), as a numpy 1d array

    P = D_orig.iloc[ :, 1 : 2].values.flatten ()
    P = numpy.array (P)

    # Box plot

    print ('STATUS:: Boxplot of P(t) ...\n')
    plt.gcf().clear()
    plt.boxplot(P)
    plt.legend (['P(t)'])
    plt.savefig ('boxplot.png')

    # Histogram

    print ('STATUS:: Histogram of P(t) ...\n')
    plt.gcf().clear()
    plt.hist(P, 10)
    plt.legend (['P(t)'])
    plt.savefig ('histogram.png')

    print ('STATUS:: Plotting P(t) vs t ...\n')
    
    plt.gcf().clear()
    plt.plot (P)
    plt.legend (['P(t)'])
    plt.xlabel ('t (t=0 => Jan 1949)')
    plt.ylabel ('Air Passengers (in thousands)')
    plt.savefig ('P_of_t.png')
    
    # Log

    print ('STATUS:: Computing ln{P(t)} ...\n')
    ln_P = numpy.log (P)
    
    # Normalize to 0...1

    print ('STATUS:: Normalizing ln{P(t))} to 0...1\n')
    try:
        scaler = MinMaxScaler().fit (ln_P)
        norma_ln_P = scaler.transform (ln_P)
    except Exception as e:
        print (str (e))
        print ('ERROR:: Could not normalize ln{P(t)} to 0...1')
        return False

    print ('STATUS:: Plotting normaized version of ln{P(t)} vs t ...\n')
    
    plt.gcf().clear()
    plt.plot (norma_ln_P)
    plt.legend (['Normalized ln{P(t)}'])
    plt.xlabel ('t (t=0 => Jan 1949)')
    plt.savefig ('norma_ln_P_of_t.png')
    
    # Split into train and test sets

    train_x        = norma_ln_P[  0 :  96]
    test_x         = norma_ln_P[ 96 : 108]
    
    train_y        = norma_ln_P[ 0+FORECAST_HORIZON :  96+FORECAST_HORIZON]
    correct_test_y = norma_ln_P[96+FORECAST_HORIZON : 108+FORECAST_HORIZON]

    if len (train_x) != len (train_y) :
        print ('ERROR:: Incompatible lengths in training')
        return False

    # Initialize Keras's LSTM model

    print ('STATUS:: Initializing keras model ...\n')
    
    model = keras.models.Sequential ()

    if type (model) is not keras.models.Sequential:
        print ('ERROR:: Could not create a keras model')
        return False

    # Add LSTM layer
    
    print ('STATUS:: Adding LSTM layer to keras model ...\n')
    try:
        layer = keras.layers.recurrent.LSTM (units = NUM_HIDDEN_NEURONS,
                                             input_shape = (NUM_TIME_POINTS_PER_SUB_VECTOR, 1),
                                             return_sequences = True)
        model.add (layer)
    except Exception as e:
        print (str (e))
        print ('ERROR:: Could not add LSTM layers to model')
        return False

    # Add TimeDistributed Dense layer
    
    print ('STATUS:: Adding TimeDistributed Dense layer to keras model ...\n')
    try:
        layer = keras.layers.core.Dense (units = 1)
        layer = keras.layers.wrappers.TimeDistributed (layer)
        model.add (layer)
    except Exception as e:
        print (str (e))
        print ('ERROR:: Could not add TimeDistributed Dense layer to model')
        return False

    # Add Activation

    print ('STATUS:: Adding activation layer to keras model ...\n')
    try:
        layer = keras.layers.Activation (ACTIVATION)
        model.add (layer)
    except Exception as e:
        print (str (e))
        print ('ERROR:: Could not add Activation layer to model')
        return False
        
    # Compile model
    
    try:
        print ('STATUS:: Compiling keras model ...\n')
        model.compile (loss = 'mean_squared_error', optimizer = 'adam')
    except Exception as e:
        print (str (e))
        print ('ERROR:: Could not compile model')
        return False

    # Train

    try:
        print ('STATUS:: Training model ...\n')
        
        num_sub_vectors = len (train_x) / NUM_TIME_POINTS_PER_SUB_VECTOR
        num_sub_vectors = int (math.ceil (num_sub_vectors))
        
        train_xx = train_x.reshape (num_sub_vectors,
                                    NUM_TIME_POINTS_PER_SUB_VECTOR,
                                    1)
        train_yy = train_y.reshape (num_sub_vectors,
                                    NUM_TIME_POINTS_PER_SUB_VECTOR,
                                    1)

        t0 = get_epoch_time_in_seconds ()
        model.fit (train_xx, train_yy, batch_size = 6, epochs = 50, verbose = 1)
        t1 = get_epoch_time_in_seconds ()
        elapsed_time = t1 - t0
        print ('STATUS:: Trained the model in %.3f s\n' % elapsed_time)
    except Exception as e:
        print (str (e))
        print ('ERROR:: Could not train model')
        return False

    # Predict

    try:
        print ('STATUS:: Testing model ...\n')
        
        num_sub_vectors = len (test_x) / NUM_TIME_POINTS_PER_SUB_VECTOR
        num_sub_vectors = int (math.ceil (num_sub_vectors))
        
        test_xx = test_x.reshape (num_sub_vectors,
                                  NUM_TIME_POINTS_PER_SUB_VECTOR,
                                  1)
        
        test_yy = model.predict (test_xx, batch_size = 6)
    except Exception as e:
        print (str (e))
        print ('ERROR:: Could not test model')
        return False

    test_y = test_yy.reshape (len(test_x))

    # Undo various transformations on test_y

    print ('STATUS:: Reverting out of 0...1 to original scale ...\n')
    test_y = scaler.inverse_transform (test_y)

    print ('STATUS:: Undoing log scaling ...\n')
    test_y = numpy.exp (test_y)

    # Create a duplicate name

    P_f = test_y

    print ('STATUS:: Plotting P(t) and P_f(t) vs t ...\n')
    plt.gcf().clear()
    plt.plot (P[132:144])
    plt.plot (P_f)

    title = 'Forecasts with ' + str (NUM_HIDDEN_NEURONS) + ' hidden neurons'
    plt.suptitle (title)
    plt.legend (['P(t)', 'P_f(t)'])
    plt.xlabel ('t (t=0 => Jan 1958)')
    plt.ylabel ('Air Passengers (in thousands)')

    png_file_name = 'compare_' + str (NUM_HIDDEN_NEURONS) + '.png'
    plt.savefig (png_file_name)

    # Mean Absolute Scaled Error of Hyndman & Koehler

    mase = compute_mase (P, P_f, 132, FORECAST_HORIZON)

    if mase == -1:
        print ('WARNING:: Could not compute Mean Absolute Scaled Error')
    else:
        print ('STATUS:: Mean Absolute Scaled Error : %.2f\n' % mase)
        
    print ('STATUS:: End of my_main\n')

    return True

#-------------------------------------------------------------------------------

# RETURN VALUE
# True => Code has been launched from some IDE
# False => Code is being run as a script from some XTerm

def is_interactive_mode ():
    
    try:
        
        if sys.ps1:
            return True

    except AttributeError:
        
        if sys.flags.interactive:
            return True
        else:
            return False

#-------------------------------------------------------------------------------

if __name__ == '__main__':

    n = my_main ()
    print ('\n')
    
    if is_interactive_mode ():
        quit
    elif n:
        sys.exit (0)
    else:
        sys.exit (1)

#-------------------------------------------------------------------------------
