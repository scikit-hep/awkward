import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

file=open(os.path.join(CURRENT_DIR,'..','outputs.txt'),'rb')
unique=set(file.readlines())
file.close()

file=open(os.path.join(CURRENT_DIR,'..','outputs.txt'),'wb').writelines(unique)
