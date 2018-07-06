import numpy as np
import pandas as pd
import pathlib
import operator

class typyDB(object):
    ''' A database for storing simulation data

    Attributes
    ----------
    trial_index: pd.DataFrame, size (ntrials,ntlabels)
        Table of trials and parameters describing trials. The index of this
        DataFrame should be unique as it identifies the trial in other tables. 

    data_index: pd.DataFrame, size (ndata,ndlabels)
        Table of data for all trials. Each row in the trial_index corresponds
        to many rows in this table by the trial_id column. 

    data: dict of pd.DataFrames
        dictionary keyed by data length (i.e. vector length) with DataFrames as
        values. Each DataFrame stores vector data where all vectors in a
        dataframe are the same length corresponding to the dict key.
    '''
    def __init__(self,trial_labels=None,data_labels=None,path=None,prefix=None):
        '''Constructor 
        
        User should specify either both trial_ and data_labels or both path and
        prefix. The former creates a new and empty db while the latter attempts
        to load a db from disk.
        
        Arguments:
        ----------
        trial_labels: list
            column labels for trial_index
            
        data_labels: list
            column labels for trial_index
            
        path: str
            path to directory containing pickled db
        
        prefix: str
            filename prefix of pickled db
        '''
        if (trial_labels is not None) and (data_labels is not None):
            self.trial_labels = trial_labels
            self.data_labels = data_labels
            
            self.trial_index = pd.DataFrame(columns=trial_labels)
            self.trial_index.name = 'trial_id'
            
            # We add three columns to the data_index 
            ## trial_id is for joining with trial_index
            ## vector_id is for joining with a data table
            ## length is for selecting the
            self.data_index= pd.DataFrame(columns=['trial_id','vector_id','length'] + data_labels)
            self.data = {}
        elif (path is not None) and (prefix is not None):
            self.load(path,prefix)
        else:
            err = 'Incompatible argument to db constructor.'
            err+= '\n Either specify both trial_labels and data_labels or'
            err+= '\n specify both path and prefix.'
            raise ValueError(err)
    
    def save(self,path,prefix):
        '''Save db to disk
        
        Arguments
        ---------
        path: str
            path to directory containing pickled db
        
        prefix: str
            filename prefix of pickled db
        '''
        path = pathlib.Path(path)
        
        fname = path / (prefix + '-tindex.pkl')
        self.trial_index.to_pickle(fname)
        
        fname = path / (prefix + '-dindex.pkl')
        self.data_index.to_pickle(fname)
        
        for k,v in self.data.items():
            fname = path / (prefix + '-{}-data.pkl'.format(k))
            v.to_pickle(fname)
    
    def load(self,path,prefix):
        '''load db from disk
        
        Arguments
        ---------
        path: str
            path to directory containing pickled db
        
        prefix: str
            filename prefix of pickled db
        '''
        path = pathlib.Path(path)
        
        fname = path / (prefix + '-tindex.pkl')
        self.trial_index = pd.read_pickle(fname)
        self.trial_labels = list(self.trial_index.columns)
        
        fname = path / (prefix + '-dindex.pkl')
        self.data_index = pd.read_pickle(fname)
        
        # Must strip columns added in __init__() so that add_data() can created
        # temporary DataFrame without these columns
        self.data_labels = list(self.data_index.columns)
        self.data_labels.remove('trial_id') 
        self.data_labels.remove('vector_id')
        self.data_labels.remove('length')
        
        self.data = {}
        for fname in path.glob(prefix + '-*-data.pkl'):
            k = int(str(fname).split('-')[-2])
            self.data[k] = pd.read_pickle(fname)
        
    def validate_trial_key(self,trial_key):
        '''Try to sanity check passed trial_keys.'''

        try:
            nkeys = np.shape(trial_key)[1] # list of trial dkeys
        except IndexError:
            nkeys = np.shape(trial_key)[0] # single trial dkeys

        if  nkeys != self.trial_index.shape[-1]:
            err = 'trial_key shape mismatched with trial_index'
            err += '\ntrial_key shape: {}'.format(np.shape(trial_key))
            err += '\ntrial_index shape: {}'.format(np.shape(self.trial_index.shape))
            raise ValueError(err)
    
    def get_trial(self,trial_key):
        '''Returns first row matching trial_key

        Search through trial_index and return the first row and row_number that
        matches trial_key.

        Arguments
        ---------
        trial_key: list
            row-values to search for in trial_index
            
        Returns
        -------
        trial: pd.DataFrame or None
            If a matching row is found: DataFrame of matching row
            If not found: None

        trial_id: pd.DataFrame or None
            If a matching row is found: row number of matching row
            If not found: None
        '''
        self.validate_trial_key(trial_key)

        #all columns must match
        match = (self.trial_index==np.asarray(trial_key)).all(1) 
        if match.any(): #trial found!
            #iloc[0] gets first match
            trial = self.trial_index.loc[match].iloc[0] 
            trial_id = self.trial_index.loc[match].index[0]
        else: #no trial found!
            trial = None
            trial_id = None
        return trial,trial_id
    
    def add_trials(self,trial_keys):
        '''Add several trials to trial_index

        .. note::

            Duplicates entries are removed from the trial_index using the
            DataFrame.drop_duplicates() command.

        Arguments
        ---------
        trial_keys: list of lists, or list of dict
            List of rows to be added to the trial_index. Each value in the list
            is a list of column values either as a sublist or a dictionary. 

        Example
        -------
        .. code-block:: python

            trial_labels = ['dispersity','mass','conc']
            data_labels = ['name','date','notes']
            db = typyDB(trial_labels=trial_labels,data_labels=data_labels)

            trial_keys = []
            trial_keys.append({'dispersity':1.5,'mass':2.25,'conc':0.3})
            trial_keys.append([1.25,2.25,0.3])
            trial_keys.append({'dispersity':1.45,'mass':2.15,'conc':0.2})

            db.add_trials(trial_keys)

            db.trial_index #show all trials

        '''
        self.validate_trial_key(trial_keys)

        new_trials = pd.DataFrame(trial_keys,columns=self.trial_labels)
        try:
            shift_val = max(self.trial_index.index)+1
        except ValueError:
            shift_val = 0
        new_trials.index += shift_val
            
        self.trial_index = pd.concat([self.trial_index,new_trials])
        self.trial_index.drop_duplicates(inplace=True)
    
    def add_trial(self,trial_key):
        '''Add a single trial to trial_index

        .. note::

            Duplicates entries are not added to the trial_index

        Arguments
        ---------
        trial_keys: list of lists, or list of dict
            List of rows to be added to the trial_index. Each value in the list
            is a list of column values either as a sublist or a dictionary. 

        Returns
        -------
        trial: pd.DataFrame
            DataFrame row corresponding to the added trial in the trial_index

        trial_id: int
            pandas index to the trial_index DataFrame

        Example
        -------
        .. code-block:: python

            trial_labels = ['dispersity','mass','conc']
            data_labels = ['name','date','notes']
            db = typyDB(trial_labels=trial_labels,data_labels=data_labels)

            db.add_trials([1.25,2.25,0.3)

            db.trial_index #show all trials
        '''
        self.validate_trial_key(trial_key)

        # look for trial in trial_index
        trial,trial_id = self.get_trial(trial_key)
        if trial_id is None: #need to add 
            try:
                trial_id = max(self.trial_index.index) + 1
            except ValueError:
                trial_id = 0
            self.trial_index.loc[trial_id] = trial_key
            trial = self.trial_index.loc[trial_id]
        else: # Do nothing because trial is already in trial_index
            pass
        return trial,trial_id
    
    def add_data(self,data_keys,data,trial_key=None,trial_id=None):
        if (trial_key is None) and (trial_id is None):
            raise ValueError('Must specify either trial_key or trial_id')
        elif trial_id is None:
            _,trial_id = self.get_trial(trial_key)

        if trial_id is None:
            raise ValueError('trial_id not specified or not found in trial_index')

        if len(data_keys)!=len(data):
            raise ValueError('data_keys and data do not have the same number of entries')
        
        if np.ndim(data)!=2:
            raise ValueError('data array must be 2D with ndata rows of the same column length')
            
        data = pd.DataFrame(data)
        ndata,ldata = data.shape
        if not (ldata in self.data):
            istart = 0
            self.data[ldata] = data
            self.data[ldata].index.name = 'vector_id'
        else:
            istart = self.data[ldata].shape[0]
            self.data[ldata] = pd.concat([self.data[ldata],data],ignore_index=True,sort=True)
            self.data[ldata].index.name = 'vector_id'
        
        data_index = pd.DataFrame(data_keys,columns=self.data_labels)
        data_index['trial_id'] = trial_id
        data_index['vector_id'] = np.arange(istart,istart+ndata,dtype=int)
        data_index['length'] = ldata
        self.data_index = pd.concat([self.data_index,data_index],ignore_index=True,sort=True)

        # need to make sure the trial_id column is mergable with the trial_index index
        self.data_index.trial_id = self.data_index.trial_id.astype(int)

        
    def build_mask(self,sel,index):
        '''
        Arguments
        ---------
        sel: dict
            dictionary of specifiers, where the keys of the dictionary match
            columns of the index. See below for more information.
            
        Case 1: Value of sel at a key is a str/int/float.
        Result: Rows are matched where value at in column == key is equal t the value
        
        Case 2: Value of sel at a key is a dict with two keys: op and val
        
        Case 3: Value of sel at a key is a list of values and/or dictionaries.
        
        '''
        mask = np.ones(index.shape[0],dtype=bool)
        for k,v1 in sel.items():
            # v1 needs to be iterable; hopefully no one passed an ndarray
            if not isinstance(v1,list):
                v1 = [v1]
            
            sub_mask = np.zeros_like(mask,dtype=bool)
            for v2 in v1:
                if isinstance(v2,dict):# specialized operator
                    op = v2['op']
                    val = v2['val']
                else: # assume operature.eq
                    op = operator.eq
                    val = v2
                sub_mask = np.logical_or(sub_mask,op(index[k],val))
            mask &= sub_mask
        return mask
    
    def select(self,trial_select=None,data_select=None,trial_id=None):
        '''
        Arguments
        ---------
        trial_select,data_select: int/float/str/dict or list of int/float/str/dict
            See description above
        
        '''
        if all([sel is None for sel in [trial_select,data_select,trial_id]]):
            raise ValueError('data_select and (trial_select or trial_id) must be specified.')
        elif all([sel is None for sel in [trial_select,trial_id]]):
            raise ValueError('(trial_select or trial_id) must be specified.')
        elif all([sel is not None for sel in [trial_select,trial_id]]):
            raise ValueError('Do not specify both trial_select or trial_id.')

        if trial_select is not None:
            # boolean mask for trial_index df
            trial_mask = self.build_mask(trial_select,self.trial_index)
        else:
            trial_mask = trial_id

        trial_index_sel = self.trial_index.loc[trial_mask]
        
        # left index in trial_index should correspond to trial_id column
        data_index_sel1 = trial_index_sel.merge(self.data_index,
                                                left_index=True,
                                                right_on='trial_id')
        
        # boolean mask for data_index
        data_index_mask = self.build_mask(data_select,data_index_sel1)
        data_index_sel2 = data_index_sel1.loc[data_index_mask]
        
        index = []
        data = []
        for data_len,group in data_index_sel2.groupby('length'):
            index.append(group)
            data_mask = group.vector_id
            data.append(self.data[data_len].loc[data_mask])
            
        if len(index) == 1:
            return index[0],data[0]
        else:
            return index,data
            
