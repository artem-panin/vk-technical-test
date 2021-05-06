### Исходные данные: 

Aнонимизированный датасет пользователей, где для каждого пользователя указан список текущих друзей пользователя. Для каждых двух пользователей известно, как давно они подружились и насколько активно взаимодействуют в соц. сети, обе величины измеряются в некоторых абстрактных величинах.


### Исходная задача: 

Cделать систему рекомендации друзей в социальной сети, которая для данного пользователя выдает список пользователей, с кем пользователь вероятней всего подружится.

1) Придумайте, как переформулировать продуктовую задачу в Data Science задачу. 
2) Сделайте два варианта работы алгоритма:
* через эвристику, не требующую машинного обучения
* через любой близкий вам вариант машинного обучения – обучение с учителем, рекомендательные системы, итд.
* эвристика будет вашим бейзлайн-алгоритмом. Докажите, что ваш ML алгоритм лучше решает поставленную задачу.

Данные три пункта необходимо оформить в виде Jupiter Notebook, с вашими комментариями, почему вы приняли то или иное решение в ходе выполнения задания.

3) Напишите утилиту (это может быть программа на python или docker контейнер, в зависимости от ваших предпочтений), которая будет эмулировать работу вашего алгоритма в production. Программа должна отвечать на запросы за адекватное время. Программа интерактивно принимает на вход по одному идентификатору пользователя на каждой строке и для каждой строки из ввода возвращает 5 пользователей, с которыми данный пользователь вероятней всего подружится.


### Описание решения: 

* директория data содержит в себе начальные данные и предобработанные (для валидации: в директории validated, для итоговой модели: в директории prod)
* Jupyter-ноутбуки (heuristics_approach, machine_learning_approach, models_comparison) содержат описания решения первоначальных задач и сравнение полученных результатов
* preprocessing.py - модуль для предобработки и кэширования данных
* recommender_system.py - реализация классов 3х различных моделей (HeuristicsModel, CollaborativeFilteringModel, KNNModel)
* metrics.py - скрипт с метриками для оценивания
* emulator.py - эмулятор продакшн-версии модели
* requirements.txt - описание зависимостей проекта