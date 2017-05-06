El archivo FileCreator.py crea los conjuntos de entrenamiento y prueba al azar a partir de los datos guardados en el
archivo sensorless_tarea2.txt y los guarda en los archivos training.txt y tester.txt respectivamente.

Hay ocasiones en que la función random.shuffle elimina los elementos del arreglo por algún motivo, por lo que para bajos
tasas de error (<=0.02) puede ser que se necesite ejecutar varias veces el archivo.

 Para la ejecución del programa principal está el archivo Main.py, aquí, la creación y el testeo del clasificador se
 realiza en el constructor, por lo que al final del archivo está la construcción de uno de estos objetos.

 En el constructor se asume la existencia de los archivos training.txt y tester.txt para la creación de los conjuntos
 de entrenamiento y prueba.