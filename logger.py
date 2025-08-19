
import pandas as pd 
import numpy as np
from pathlib import Path

class Logger:
    
    def __init__ (self, log_template, log_path, logprint: bool = False):
        '''
        Class to log all experimental runs data
        
        Parameters
        ----------
        log_template : pd.DataFrame
            a template of the log table 
        log_path : f-str
            where to keep the log table  
        logprint : bool 
            needs to log in console
        '''
        self.log_template = log_template
        self.log_path = log_path 
        self.logprint = logprint
    
    def create_logfile (self):
    

        '''
        Create an empty pandas log table from setted template
        Parameters 
        ----------
        name : str 
            the name of the log table
        '''

        log_file = self.log_template.copy() 

        if (not Path(self.log_path).exists()):
            log_file.to_csv(self.log_path, sep = ',', index = False)
            return 0
        else:
            print (f'Logfile {self.log_path} already exists')
            return 1
        
    def _update_columns (self): 
        ''' 
        Update columns' names according to the template with no data changes 
        '''

        log_file = pd.read_csv(self.log_path)
        log_file.columns = self.log_template.columns
        log_file.to_csv(self.log_path, sep = ',', index = False)
    
    def _remove_lines_by_id (self, ids : list[int]): 
        ''' 
        Delete lines with listed ids from logfile

        Parameters
        ----------
            ids : list[int]
        ''' 

        log_file = pd.read_csv(self.log_path)
        for id in ids: 
            log_file = log_file.drop(log_file.index[log_file['Id (int)'] == id])
        log_file.to_csv(self.log_path, sep = ',', index = False)
    

    def _load_logfile (self) -> [pd.DataFrame, int]: 

        '''
        Get a pandas dataframe and the last line index

        Returns
        -------
        log_file : pd.DataFrame 
            a dataframe of a logfile 
        id : int 
            index of the last log line
        '''

        log_file = pd.DataFrame()
        id = 0
        try: 
            log_file = pd.read_csv(self.log_path)

            if (list(log_file.columns) != list(self.log_template.columns)):
                print(f"Columns in {self.log_path} are broken. Expected {self.log_template.columns} and got {log_file.columns} \n missmatch: {set(log_file.columns) - set(self.log_template.columns)}")
            else: 
                if (self.logprint): print(f"Successfully loaded {self.log_path} with content")
                if (not log_file.empty):
                    id = int (log_file.iloc[len(log_file) - 1]['Id (int)'])
                else:
                    id = 0
                    print('Log_file does not have a buffer line. So id is broken')
        except FileNotFoundError: 
            print (f'There is no logfile {self.log_path}')
    

        return log_file, id 

    def _matrix_to_str (self, matrix: np.ndarray) -> str: 

        string = '['
        for i in range (matrix.shape[0]):
            string += '['
            for j in range (matrix.shape[1]):
                string += f"{matrix[i][j]}"
                if (j < matrix.shape[1] - 1): string += ','
            string += ']'
            if (i < matrix.shape[0] - 1): string += ','
        string += ']'
        return string 
    

    def _str_to_matrix (self, string: str) -> np.ndarray: 

        lines = string.split('],[')
        matrix = []
        for line in lines: 
            line = line.replace(']', '')
            line = line.replace('[', '')

            elements = np.array(line.split(','), dtype = float)
            matrix.append(elements)

        matrix = np.array(matrix)    
        return matrix 

    def _vector_to_str (self, vector: np.ndarray) -> str:  
        string  = ','.join(str(vector).split())
        string = string[:1] + string[1:]
        string = string.replace(',,', ',')
        string = string.replace('[,', '[')
        string = string.replace(',]', ']')
        return string

    def _str_to_vector (self, string: str) -> np.ndarray: 
        string = string.replace(']', '')
        string = string.replace('[', '')
        vector = np.array(string.split(','), dtype = float)
        return vector 

    def _line_to_data (self, Id: int) -> list: 
        
        '''Line of a csv matrix-log table out-parser
        Pameters
        ---------
        dataset: pd.DataFrame
        Id: int
            line's Id (not index!)

        Returns
        -------
        data: dict
        '''

        log_file = self._load_logfile()[0]

        columns = self.log_template.columns

        if (len(log_file.loc[log_file['Id (int)'] == Id]) > 1): 
            print('Warning: there are two lines with similar ids')
    
        line = log_file.loc[log_file['Id (int)'] == Id].to_dict()
        
        data = {key: 0 for key in columns}
        try:
            for key in columns: 
                #print(line[key])
                
                val = line[key][list(line[key].keys())[0]]
                #print(type(val))
                if (val != 'None') and (str(type(val)) != '<class \'NoneType\'>'):
                    if (key[-5:] == '(str)'):
                        try:
                            data[key] = val
                        except: pass
                    if (key[-5:] == '(int)'):
                        try:
                            data[key] = int(val)
                        except: pass
                    if (key[-5:] == '(flt)'):
                        try:
                            data[key] = float(val)
                        except: pass 
                    if (key[-5:] == '(vec)'):
                        try:
                            data[key] = self._str_to_vector(val)
                        except: pass
                    if (key[-5:] == '(mat)'):
                        try:
                            data[key] = self._str_to_matrix(val)
                        except:pass
        except: 
            print('No line with id {id} - there is nothing to read')
        return data

    def _data_to_line (self, data: dict, byid: bool = False, Id: int = 0):
        
        '''
        Convert data dict to a log_file line 

        Parameters 
        ----------
        data : pd.DataFrame
            a dictionary with data, dictionary keys are log_file's columns
        byid: bool 
            id of a line to log in
        '''

        columns = self.log_template.columns
        line = {}

        log_file = self._load_logfile()[0]
        rend_id = len(log_file)
        for key in columns: 

            val = data[key]
            if (key[-5:] == '(str)'):
                line[key] = val
            if (key[-5:] == '(int)'):
                line[key] = val
            if (key[-5:] == '(flt)'):
                line[key] = val
            if (key[-5:] == '(vec)'):
                line[key] = self._vector_to_str(val)
            if (key[-5:] == '(mat)'):
                line[key] = self._matrix_to_str(val)

        if (byid):
            log_file.loc[log_file['Id (int)'] == Id, list(line.keys())] = list(line.values())
        else:  
            log_file = pd.concat(
            [log_file, pd.DataFrame([line])],
            ignore_index=True
            )
        log_file.to_csv(self.log_path, sep = ',', index = False)

    def log (self, data: dict, byid: bool = False, Id: int = -1):
        
        self._data_to_line(data = data, byid = byid, Id = Id)
        
    
    def read (self, id):
        return self._line_to_data(Id = id)
    

