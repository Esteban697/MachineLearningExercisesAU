# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:58:18 2018

@author: esteb
"""

def convert(name1,name2):
    #File input
    fileInput = open(name1+".txt","r")
    
    #File output
    fileOutput = open(name2+".fasta","w")
    
    #Seq count
    count = 1 ;
    
    #Loop through each line in the input file
    print("Converting to FASTA...")
    for strLine in fileInput:
    
        #Strip the endline character from each input line
        strLine = strLine.rstrip("\n")
    
        #Output the header
        fileOutput.write(">" + str(count) + "\n")
        fileOutput.write(strLine + "\n")
    
        count = count + 1
    print("Done.")
    
    #Close the input and output file
    fileInput.close()
    fileOutput.close()