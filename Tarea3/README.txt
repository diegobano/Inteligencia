Para ejecutar los distintos programas:

FileCreator.py al ejecutarse lee el archivo sensorless_tarea2.txt dentro de la misma ubicación donde se encuentra y lo
divide en dos archivos de entrenamiento y prueba (training.txt y tester.txt respectivamente)

Dado que la lejanía que se espera entre el conjunto de entrenamiento y el conjunto original es baja, suele necesitar
ejecutarse más de una vez porque el método random.shuffle no funciona bien y se queda iterando por siempre, en tal caso
hay que cancelar y volver a ejecutar.

El archivo Main.py hace todo en su constructor, para ejecutarlo basta crear un objeto de la clase al final del archivo
y se van a leer los archivos de entrenamiento y prueba antes mencionados, creando los clasificadores y todos sus
gráficos por separado, uno por uno. Esto se demora unos 30 segundos en ejecutarse y genera 23 graficos en total, uno
para el kernel lineal, y 11 para cada uno de los otros.