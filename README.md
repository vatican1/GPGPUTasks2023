В этом репозитории предложены задания для курса по вычислениям на видеокартах 2023

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2023/).

# Задание 7. Radix sort

[![Build Status](https://github.com/GPGPUCourse/GPGPUTasks2023/actions/workflows/cmake.yml/badge.svg?branch=task07&event=push)](https://github.com/GPGPUCourse/GPGPUTasks2023/actions/workflows/cmake.yml)

0. Сделать fork проекта
1. Выполнить задание 7.1
2. Отправить **Pull-request** с названием ```Task07 <Имя> <Фамилия> <Аффиляция>``` (указав вывод каждой программы при исполнении на вашем компьютере - в тройных кавычках для сохранения форматирования)

**Дедлайн**: 23:59 29 октября.


Задание 7.1. Radix sort
=========

Реализуйте radix sort для unsigned int (используя локальную память).

Не влияет на баллы, но вероятно, что будет интересно сравнить новую сортировку с остальными по скорости. В случае, если вы используете линейный алгоритм для префиксной суммы, то radix sort тоже становится линейным и должен обгонять merge и bitonic sort начиная с какого-то размера массива.

Файлы: ```src/main_radix.cpp``` и ```src/cl/radix.cl```
