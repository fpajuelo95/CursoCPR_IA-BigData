#!/usr/bin/python3
import sys 
for linea in sys.stdin: 
    # eliminamos los espacios de delante y de detrás
    linea = linea.strip() 
    # dividimos la línea en palabras
    palabras = linea.split() 
    # creamos tuplas de (palabra, 1)
    for palabra in palabras: 
        print(palabra, "\t1")