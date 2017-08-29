import win32com.client as comclt
import numpy as np
'''assumes ml_output is a length 2 vector where a max value at [0] signifies
staying and a max value at [1] signifies jumping'''
def press_key(ml_output):
    if ml_output == 1:
        wsh = comclt.Dispatch("WScript.Shell")
        wsh.SendKeys('{UP}')
