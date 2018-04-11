import unittest
import numpy as np
from typyDB.typyDB import typyDB

class typyDB_TestCase(unittest.TestCase):
    def create_empty_db(self):
        trial_labels = ['Nbb','nsc','Nsc','T','npoly']
        data_labels = ['name','site1','site2']
        db = typyDB(trial_labels=trial_labels,data_labels=data_labels)
        return db

    def create_filled_db(self):
        db = self.create_empty_db()

        trial_key = []
        trial_key.append([1,2,3,4,5])
        trial_key.append([2,3,4,5,1])
        trial_key.append([3,4,5,1,2])
        trial_key.append([3,4,5,1,2])
        trial_key.append([4,5,1,2,3])
        db.add_trials(trial_key)

        data_keys = []
        data = []
        data_keys.append(['omega','A','B'])
        data.append(np.ones(16))
        data_keys.append(['omega','B','B'])
        data.append(np.ones(16)*2)
        data_keys.append(['omega','A','A'])
        data.append(np.ones(16)*3)
        db.add_data(data_keys,data,trial_key=[1,2,3,4,5])

        data_keys = []
        data = []
        data_keys.append(['omega','A','B'])
        data.append(np.ones(16)*4)
        data_keys.append(['omega','B','B'])
        data.append(np.ones(16)*5)
        data_keys.append(['omega','A','A'])
        data.append(np.ones(16)*6)
        trial,trial_id = db.get_trial([4,5,1,2,3])
        db.add_data(data_keys,data,trial_id=trial_id)


        return db

    def test_create_db(self):
        ''' Can we create typyDB instance?'''
        db = self.create_empty_db()
        
    def test_add_trial(self):
        ''' Can we fill db one by one?'''
        db = self.create_empty_db()

        trial_key = [1,2,3,4,5]
        db.add_trial(trial_key)
        self.assertTupleEqual(db.trial_index.shape,(1,5))
        self.assertTupleEqual(db.data_index.shape,(0,6))
        self.assertDictEqual(db.data,{})

        # addition should fail if trial_key shape is wrong
        with self.assertRaises(ValueError):
            trial_key = [2,3,4,5]
            db.add_trial(trial_key)
        self.assertTupleEqual(db.trial_index.shape,(1,5))
        self.assertTupleEqual(db.data_index.shape,(0,6))
        self.assertDictEqual(db.data,{})

        # addition should fail silently trial_key shape is already in db
        trial_key = [1,2,3,4,5]
        db.add_trial(trial_key)
        self.assertTupleEqual(db.trial_index.shape,(1,5))
        self.assertTupleEqual(db.data_index.shape,(0,6))
        self.assertDictEqual(db.data,{})

        # addition should fail silently trial_key shape is already in db
        trial_key = [2,3,4,5,1]
        db.add_trial(trial_key)
        self.assertTupleEqual(db.trial_index.shape,(2,5))
        self.assertTupleEqual(db.data_index.shape,(0,6))
        self.assertDictEqual(db.data,{})


    def test_add_trials(self):
        ''' Can we fill db?'''
        db = self.create_empty_db()

        trial_key = []
        trial_key.append([1,2,3,4,5])
        trial_key.append([2,3,4,5,1])
        trial_key.append([3,4,5,1,2])
        trial_key.append([3,4,5,1,2])
        trial_key.append([4,5,1,2,3])
        db.add_trials(trial_key)
        self.assertTupleEqual(db.trial_index.shape,(4,5))
        self.assertListEqual(db.trial_index.index.tolist(),[0,1,2,4])
        self.assertTupleEqual(db.data_index.shape,(0,6))
        self.assertDictEqual(db.data,{})

        trial_key = []
        trial_key.append([2,2,3,4,5])
        trial_key.append([3,3,4,5,1])
        trial_key.append([4,4,5,1,2])
        trial_key.append([4,4,5,1,2])
        trial_key.append([5,5,1,2,3])
        db.add_trials(trial_key)
        self.assertTupleEqual(db.trial_index.shape,(8,5))
        self.assertListEqual(db.trial_index.index.tolist(),[0,1,2,4,5,6,7,9])
        self.assertTupleEqual(db.data_index.shape,(0,6))
        self.assertDictEqual(db.data,{})

    def test_add_data(self):
        '''Can we add data to db?'''
        db = self.create_empty_db()

        trial_key = []
        trial_key.append([1,2,3,4,5])
        trial_key.append([2,3,4,5,1])
        trial_key.append([3,4,5,1,2])
        trial_key.append([3,4,5,1,2])
        trial_key.append([4,5,1,2,3])
        db.add_trials(trial_key)

        data_keys = []
        data = []
        data_keys.append(['omega','A','B'])
        data.append(np.ones(16))
        data_keys.append(['omega','B','B'])
        data.append(np.ones(16)*2)
        data_keys.append(['omega','A','A'])
        data.append(np.ones(16)*3)

        db.add_data(data_keys,data,trial_key=[1,2,3,4,5])
        self.assertTupleEqual(db.trial_index.shape,(4,5))
        self.assertListEqual(db.trial_index.index.tolist(),[0,1,2,4])
        self.assertTupleEqual(db.data_index.shape,(3,6))
        self.assertTupleEqual(db.data[16].shape,(3,16))
        self.assertListEqual(db.data_index['trial_id'].tolist(),[0,0,0])

        trial,trial_id = db.get_trial([4,5,1,2,3])
        db.add_data(data_keys,data,trial_id=trial_id)
        self.assertTupleEqual(db.trial_index.shape,(4,5))
        self.assertListEqual(db.trial_index.index.tolist(),[0,1,2,4])
        self.assertTupleEqual(db.data_index.shape,(6,6))
        self.assertTupleEqual(db.data[16].shape,(6,16))
        self.assertListEqual(db.data_index['trial_id'].tolist(),[0,0,0,4,4,4])

    def test_select1(self):
        '''Test basic selection'''
        db = self.create_filled_db()

        trial_sel = {}
        trial_sel['Nbb'] = 1
        data_sel = {}
        data_sel['site1'] = 'A'

        index ,data = db.select(trial_sel,data_sel)

        self.assertTupleEqual(data.shape,(2,16))

    def test_select2(self):
        '''Test OR selection'''
        db = self.create_filled_db()

        trial_sel = {}
        trial_sel['Nbb'] = [1,4]
        data_sel = {}
        data_sel['site1'] = 'A'

        index ,data = db.select(trial_sel,data_sel)
        self.assertTupleEqual(data.shape,(4,16))

        np.testing.assert_array_almost_equal(np.array(data.iloc[0]),np.ones(16)*1)
        np.testing.assert_array_almost_equal(np.array(data.iloc[1]),np.ones(16)*3)
        np.testing.assert_array_almost_equal(np.array(data.iloc[2]),np.ones(16)*4)
        np.testing.assert_array_almost_equal(np.array(data.iloc[3]),np.ones(16)*6)

    def test_select3(self):
        '''Test multiple data tables'''
        db = self.create_filled_db()


        data_keys = []
        data = []
        data_keys.append(['omega','A','B'])
        data.append(np.ones(8)*4)
        data_keys.append(['omega','B','B'])
        data.append(np.ones(8)*5)
        data_keys.append(['omega','A','A'])
        data.append(np.ones(8)*6)
        trial,trial_id = db.get_trial([4,5,1,2,3])
        db.add_data(data_keys,data,trial_id=trial_id)

        trial_sel = {}
        trial_sel['Nbb'] = [1,4]
        data_sel = {}
        data_sel['site1'] = 'A'

        (index1,index2),(data1,data2) = db.select(trial_sel,data_sel)

        if data1.shape[1] == 16:
            index16 = index1
            data16 = data1
            index8 = index2
            data8 = data2
        else:
            index16 = index2
            data16  = data2
            index8  = index1
            data8   = data1

        self.assertTupleEqual(data16.shape,(4,16))
        self.assertTupleEqual(data8.shape,(2,8))

        np.testing.assert_array_almost_equal(np.array(data16.iloc[0]),np.ones(16)*1)
        np.testing.assert_array_almost_equal(np.array(data16.iloc[1]),np.ones(16)*3)
        np.testing.assert_array_almost_equal(np.array(data16.iloc[2]),np.ones(16)*4)
        np.testing.assert_array_almost_equal(np.array(data16.iloc[3]),np.ones(16)*6)

        np.testing.assert_array_almost_equal(np.array(data8.iloc[0]),np.ones(8)*4)
        np.testing.assert_array_almost_equal(np.array(data8.iloc[1]),np.ones(8)*6)

    def test_select4(self):
        '''Test select by trial_id'''
        db = self.create_filled_db()

        data_sel = {}
        data_sel['site1'] = 'A'

        index ,data = db.select(trial_id=[0,4],data_select=data_sel)
        self.assertTupleEqual(data.shape,(4,16))

        np.testing.assert_array_almost_equal(np.array(data.iloc[0]),np.ones(16)*1)
        np.testing.assert_array_almost_equal(np.array(data.iloc[1]),np.ones(16)*3)
        np.testing.assert_array_almost_equal(np.array(data.iloc[2]),np.ones(16)*4)
        np.testing.assert_array_almost_equal(np.array(data.iloc[3]),np.ones(16)*6)

if __name__ == '__main__':
    import unittest 
    suite = unittest.TestLoader().loadTestsFromTestCase(typyDB_TestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
