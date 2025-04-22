''' данный код реализует оценку координат GNSS-приёмника по эфемеридам 32 спутников из навигационного послания в RINEX-формате 
с применением фильтра Калмана и визуализацией эволюции координат, ошибок, спутников и их орбит, 
с указанием "хороших" спутников (угол > 15 гр.) и геодезическим положением приёмника. '''

import math  
import pandas as pd  
import re  
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.lines import Line2D 

# константы
mu = 3.986005e14  # гравитационный параметр Земли
omega_e = 7.2921151467e-5  # угловая скорость вращения Земли
c = 299792458  # скорость света
a = 6378137.0  # большая полуось Земли (экваториальный радиус)
e2 = 0.00669437999  # квадрат эксцентриситета

# координаты приёмника в ECEF (Москва)
def get_receiver_position():
    lat = math.radians(55 + 36 / 60)  # широта в радианах
    lon = math.radians(37 + 36 / 60)  # долгота в радианах
    h = 200  # высота над уровнем моря в метрах
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)  # радиус кривизны в первом вертикале
    X = (N + h) * math.cos(lat) * math.cos(lon)  
    Y = (N + h) * math.cos(lat) * math.sin(lon)  
    Z = (N * (1 - e2) + h) * math.sin(lat) 
    return np.array([X, Y, Z]) 

def ecef_to_geodetic(X, Y, Z):  # перевод координат из ECEF в геодезические
    lon = math.atan2(Y, X)  # долгота
    p = math.sqrt(X**2 + Y**2)  # расстояние до оси вращения
    lat = math.atan2(Z, p * (1 - e2))  # начальная аппроксимация широты
    for _ in range(5):  # итеративное уточнение
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        h = p / math.cos(lat) - N  
        lat = math.atan2(Z, p * (1 - e2 * N / (N + h))) 
    return math.degrees(lat), math.degrees(lon), h  

def compute_elevation(receiver_pos, sat_pos): #расчёт угла возвышения над горизонтом
    vec = sat_pos - receiver_pos
    vec_norm = vec / np.linalg.norm(vec)
    zenith = receiver_pos / np.linalg.norm(receiver_pos)
    cos_el = np.dot(vec_norm, zenith)
    el = np.arcsin(cos_el)
    return np.degrees(el)

# чтение RINEX-файла
def parse_all_satellites(file_path):  # парсинг эфемерид по спутникам
    with open(file_path, 'r', encoding='latin-1') as file:
        lines = file.readlines()
    satellites = {}  # словарь: ID спутника -> данные эфемерид
    current_sat = None
    current_data = []
    for line in lines:
        if re.match(r'^G\d{2}', line):  # начало блока данных нового спутника
            if current_sat and current_data:
                satellites[current_sat] = current_data
            current_sat = line[:3]  # идентификатор спутника
            current_data = []
        if current_sat:
            fixed_line = re.sub(r'(\d)-', r'\1 -', line.strip())  # фиксим записи экспонент
            current_data.append(fixed_line)
    if current_sat and current_data:
        satellites[current_sat] = current_data 
    return satellites 

# расчёт орбит спутников
def compute_satellite_orbit(eph, duration=86400, step=600):  # вычисляем орбиту на сутки вперёд с шагом 600 сек (10 минут)
    try:
        mean_motion_delta = float(eph[1].split()[2].replace('D', 'E'))  # изменение среднего движения спутника
        mean_anomaly_at_epoch = float(eph[1].split()[3].replace('D', 'E'))  # средняя аномалия в момент эпохи
        eccentricity = float(eph[2].split()[1].replace('D', 'E'))  # эксцентриситет орбиты
        semi_major_axis_sqrt  = float(eph[2].split()[3].replace('D', 'E'))  # квадратный корень из большой полуоси
        time_of_ephemeris = float(eph[3].split()[0].replace('D', 'E'))  # время эпохи эфемерид
        perigee_argument = float(eph[4].split()[2].replace('D', 'E'))  # аргумент перигея
        right_ascension_at_epoch = float(eph[3].split()[2].replace('D', 'E'))  # прямое восхождение (долгота) восходящего узла в момент эпохи
        right_ascension_rate = float(eph[4].split()[3].replace('D', 'E'))  # скорость изменения прямого восхождения
        inclination_at_epoch = float(eph[4].split()[0].replace('D', 'E'))  # наклонение орбиты в момент эпохи
        inclination_rate = float(eph[5].split()[0].replace('D', 'E'))  # скорость изменения наклонения орбиты

        A = semi_major_axis_sqrt ** 2  # большая полуось (в метрах)
        n0 = math.sqrt(mu / A**3)  # среднее движение без возмущений
        n = n0 + mean_motion_delta  # поправленное среднее движение

        coords = []
        for dt in range(0, duration, step):  # вычисляем координаты на каждом шаге времени
            t = time_of_ephemeris + dt
            M = mean_anomaly_at_epoch + n * (t - time_of_ephemeris)  # средняя аномалия
            E = M  # начальное приближение эксцентрической аномалии
            for _ in range(10):
                E = M + eccentricity * math.sin(E)  # уточнение эксцентрической аномалии итерациями
            nu = 2 * math.atan2(math.sqrt(1 - eccentricity**2) * math.sin(E / 2), math.sqrt(1 + eccentricity) * math.cos(E / 2))  # истинная аномалия
            u = nu + perigee_argument  # аргумент широты (сумма истинной аномалии и аргумента перигея)
            r = A * (1 - eccentricity * math.cos(E))  # расстояние от центра Земли до спутника
            i = inclination_at_epoch + inclination_rate * (t - time_of_ephemeris)  # наклонение орбиты на текущий момент
            x_prime = r * math.cos(u)  # координата в плоскости орбиты (x)
            y_prime = r * math.sin(u)  # координата в плоскости орбиты (y)
            Omega = right_ascension_at_epoch + (right_ascension_rate - omega_e) * (t - time_of_ephemeris) - omega_e * time_of_ephemeris  # долгота восходящего узла с учётом вращения Земли
            Xs = x_prime * math.cos(Omega) - y_prime * math.cos(i) * math.sin(Omega)  # преобразование в систему ECEF: координата X
            Ys = x_prime * math.sin(Omega) + y_prime * math.cos(i) * math.cos(Omega)  # координата Y
            Zs = y_prime * math.sin(i)  # координата Z
            coords.append((Xs, Ys, Zs))  # добавляем точку траектории
        return np.array(coords)  
    except Exception:
        return None  

# реализация фильтра Калмана для оценки координат приёмника
def kalman_filter(receiver_pos, sat_positions, pseudoranges):
    X = np.append(receiver_pos + np.random.normal(0, 1000, 3), 0.001)  # начальное состояние (с шумом)
    P = np.eye(4) * 1e8  # начальная ковариационная матрица
    Q = np.eye(4) * 1e-2  # шум процесса
    R = np.eye(len(pseudoranges)) * 10  # шум измерений
    trajectory = []  # список траекторий
    residuals = []  # невязки
    for _ in range(10): 
        H, y = [], []  # матрица наблюдений и вектор ошибок
        for i, sat in enumerate(sat_positions):
            rho_est = np.linalg.norm(sat - X[:3]) + X[3] * c  # оценка дальности
            H_row = np.zeros(4)
            H_row[:3] = -(sat - X[:3]) / np.linalg.norm(sat - X[:3])  # производные по координатам
            H_row[3] = c  # производная по времени
            H.append(H_row)
            y.append(pseudoranges[i] - rho_est)
        H = np.array(H)
        y = np.array(y)
        S = H @ P @ H.T + R  # ковариация предсказания
        try:
            S_inv = np.linalg.inv(S)  # обратная матрица
        except np.linalg.LinAlgError:
            print('Предупреждение: матрица S сингулярна, использую псевдообратную')
            S_inv = np.linalg.pinv(S)
        K = P @ H.T @ S_inv  # Калмановское усиление
        X = X + K @ y  # обновление состояния
        P = (np.eye(4) - K @ H) @ P + Q  # обновление ковариации
        trajectory.append(X.copy())  # сохранение состояния
        residuals.append(np.linalg.norm(y))  # сохранение невязки
    return X, trajectory, residuals, P

# основной блок
if __name__ == '__main__': 
    rinex_path = 'Brdc0020.25n'
    receiver_position = get_receiver_position()  # расчёт координат приёмника
    satellites = parse_all_satellites(rinex_path)  # парсинг эфемерид спутников
    sat_positions = []  # список позиций спутников
    pseudoranges = []  # псевдодальности
    sat_ids = []  # идентификаторы спутников с углами
    sat_colors = []  # цвета для отображения спутников
    sat_orbits = {}  # орбиты спутников для визуализации

    for sat_id, eph in satellites.items():  # обход всех спутников
        orbit = compute_satellite_orbit(eph)  # расчёт орбиты
        if orbit is None:
            continue 
        pos = orbit[0] 
        vector = pos - receiver_position  # вектор спутник-приёмник
        distance = np.linalg.norm(vector)  # расстояние
        angle = compute_elevation(receiver_position, pos) #угол возвышения над горизонтом

        if angle < 15:
            color = 'black'
        elif angle < 30:
            color = 'yellow' 
        else:
            color = 'green'  

        sat_positions.append(pos) 
        pseudoranges.append(distance + c * 0.001)  
        sat_ids.append(f"{sat_id} ({angle:.1f}°)")  
        sat_colors.append(color)  
        sat_orbits[sat_id] = orbit 

    estimated, trajectory, residuals, P = kalman_filter(receiver_position, sat_positions, pseudoranges)  

    df = pd.DataFrame(sat_positions, columns=['X (м)', 'Y (м)', 'Z (м)'])  # таблица с координатами спутников
    df['Satellite'] = sat_ids  
    df.to_csv('visible_satellite_positions.csv', index=False) 

    fig = plt.figure(figsize=(12, 9))  
    ax = fig.add_subplot(111, projection='3d') 
    sat_positions_np = np.array(sat_positions) 
    ax.scatter(sat_positions_np[:, 0], sat_positions_np[:, 1], sat_positions_np[:, 2], c=sat_colors, alpha=0.7)  
    ax.scatter(*receiver_position, color='red', label='Приёмник')  

    for sat_id, orbit in sat_orbits.items():
        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], alpha=0.3) 

    # визуализация Земли (эллипсоид)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = a * np.outer(np.sin(u), np.sin(v))
    z = a * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, zorder=0)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Отличный спутник (>30 гр.)', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Удовлетворительный спутник (15–30 гр.)', markerfacecolor='yellow', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Плохой спутник (<15 гр.)', markerfacecolor='black', markersize=8),
        Line2D([0], [0], marker='o', color='red', label='Приёмник', markersize=8)
    ]
    ax.legend(handles=legend_elements)
    ax.set_title('Орбиты и расположение спутников в ECEF')
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_zlabel('Z (м)')
    plt.tight_layout()
    plt.show()

    # графики траектории по итерациям ФК
    traj = np.array(trajectory)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    labels = ['X', 'Y', 'Z', 'dT']
    for i, ax in enumerate(axs.flat):
        ax.plot(range(len(traj)), traj[:, i], label=labels[i])
        ax.set_xlabel('Итерация')
        ax.grid(True)
        ax.legend()
    plt.suptitle('Эволюции координат приёмника')
    plt.tight_layout()
    plt.show()

    # график невязок
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(residuals)), residuals)
    plt.title('Невязка по итерациям')
    plt.xlabel('Итерация')
    plt.ylabel('Невязка')
    plt.grid(True)
    plt.show()

    # корреляционная матрица ошибок
    corr = np.zeros_like(P)
    for i in range(4):
        for j in range(4):
            corr[i, j] = P[i, j] / (np.sqrt(P[i, i] * P[j, j]))

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title('Корреляционная матрица ошибок')
    plt.xticks(np.arange(4), ['X', 'Y', 'Z', 'dT'])
    plt.yticks(np.arange(4), ['X', 'Y', 'Z', 'dT'])
    plt.show()

    # финальный вывод координат
    lat, lon, h = ecef_to_geodetic(*estimated[:3])
    print('\nОценённые координаты приёмника:')
    print(f"X: {estimated[0]:.3f} м")
    print(f"Y: {estimated[1]:.3f} м")
    print(f"Z: {estimated[2]:.3f} м")
    print(f"Временная поправка: {estimated[3]:.9f} с")
    print(f"Широта: {lat:.6f}°   Долгота: {lon:.6f}°   Высота: {h:.2f} м")
    if 55.4 < lat < 55.8 and 37.3 < lon < 37.9:
        print("Местоположение определено верно: г. Москва, Россия")
    else:
        print("Внимание: возможное отклонение от координат г. Москвы")
