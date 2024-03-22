import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

left = -0.3;
right = 0.3;
step = 5;
def f(x):
    return x**2 + 3*np.log(x+4);
    #return np.tan(np.sin(x));
    #return np.sin(np.cos(x))#np.sqrt(np.tan(x)**2 + 7*x**2)

def plot_function(func, x_range, label, _type = '-'):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = func(x)
    
    plt.plot(x, y, _type, label=label)
    #plt.title(title)
    plt.grid(True)

def linear_func(a: float, b: float):
    def f(x):
        return a*x + b;
    return f;

def quadratic_func(a: float, b: float, c: float):
    def f(x):
        return a*x**2 + b*x + c;
    return f;

def cubic_func(a: float, b: float, c: float, d:float, xi: float):
    def f(x):
        return a*(x-xi)**3 + b*(x-xi)**2 + c*(x-xi) + d;
    return f;

#x = np.linspace(left, right, step)
#y = f(x)
#linear_spline = interp1d(x, y)
#cubic_spline = interp1d(x, y, kind='cubic')
# print(f"h = {(right - left)/(step-1)}")
# print("Расчет погрешностей: ")
# for i in range(1, len(x)):
#     x_mid = (x[i] + x[i-1]) / 2
#     y_true_mid = f(x_mid)
#     y_linear_mid = linear_spline(x_mid)
#     y_cubic_mid = cubic_spline(x_mid)
#     error_linear = abs(y_linear_mid - y_true_mid)
#     error_cubic = abs(y_cubic_mid - y_true_mid)
#     print(f"Отрезок [{x[i-1]:.4f} {x[i]:.4f}]\n\tлинейный сплайн: {error_linear:.4f}\n\tкубический сплайн: {error_cubic:.4f}")

#x_dense = np.linspace(-0.3, 0.3, 1000)
##y_true = f(x_dense)
#y_linear = linear_spline(x_dense)
#y_cubic = cubic_spline(x_dense)

_x = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5];
_y = [f(x) for x in _x];


def plot_func():
    xs = np.linspace(_x[0], _x[-1], 100);
    ys = f(xs);
    plt.plot(xs, ys, color='black')
    
def plot_linear_spline():
    #plt.plot(x_dense, linear_spline(x_dense), color='blue')
    print("Линейный сплайн: ")
    spline = [];
    for i in range(len(_x) - 1):
        a = f(_x[i]);
        b = (f(_x[i+1]) - f(_x[i]))/(_x[i+1] - _x[i]);
        a -= _x[i]*b;
        spline.append(linear_func(b, a));

    for i in range(len(spline)):
        x = np.linspace(_x[i], _x[i+1])
        _y = spline[i](x);
        plt.plot(x, _y);
        
        err = 0;
        cur_x = _x[i];
        ctr = 0;
        while cur_x < _x[i+1]:
            cur_x += 0.1;
            err += (spline[i](cur_x) - f(cur_x))**2;
            ctr += 1;
        err /= ctr;
        err **= 1/2;
        print(f'\t Отрезок [{_x[i]}, {_x[i+1]}]: средняя квадратичная погрешность: {err}')
    plt.show();

    
def half_diff(f, x1, x2):
    return (f(x1)-f(x2))/(x1-x2);

def plot_cubic_spline():
    print('Кубический сплайн: ')
    matrix = [[0 for _ in range(len(_x))] for _ in range(len(_x))];
    for i in range(len(_x)):
        if i == 0:
            matrix[i][0] = 1;
        elif i == len(_x) - 1:
            matrix[i][-1] = 1
        else:
            matrix[i][i-1] = _x[i] - _x[i-1];
            matrix[i][i] = 2*(_x[i] - _x[i-1] + _x[i+1] - _x[i]);
            matrix[i][i+1] = _x[i+1] - _x[i];
    
    fns = [0 for _ in range(len(_x))];
    for i in range(len(_x)):
        if i == 0 or i == len(_x) - 1: fns[i] = 0;
        else: fns[i] = 3*(half_diff(f, _x[i+1], _x[i]) - half_diff(f, _x[i], _x[i-1]))
    #cs = [1 for _ in range(len(_x))];
    cs = np.linalg.solve(matrix, fns);
    cs[0] = 0;
    cs[-1] = 0;
    bs = [0 for _ in range(len(_x))];
    for i in range(len(_x)-1):
        bs[i] = (f(_x[i+1]) - f(_x[i])) / (_x[i+1] - _x[i]);
        bs[i] -= (_x[i+1] - _x[i]) * (2*cs[i] + cs[i+1])/3;

    ds = [0 for _ in range(len(_x))];
    for i in range(len(_x) - 1):
        ds[i] = (cs[i+1] - cs[i])/(3*(_x[i+1] - _x[i]));

    _as = [f(x) for x in _x];
    
    spline = [0 for _ in range(len(_x) - 1)];
    for i in range(len(_x) - 1):
        spline[i] = cubic_func(ds[i], cs[i], bs[i], _as[i], _x[i] );
        
    for i in range(len(spline)):
        x = np.linspace(_x[i], _x[i+1])
        _y = spline[i](x);
        plt.plot(x, _y);

        err = 0;
        cur_x = _x[i];
        ctr = 0;
        while cur_x < _x[i+1]:
            cur_x += 0.1;
            err += (spline[i](cur_x) - f(cur_x))**2;
            ctr += 1;
        err /= ctr;
        err **= 1/2;
        print(f'\t Отрезок [{_x[i]}, {_x[i+1]}]: средняя квадратичная погрешность: {err}')
    plt.show();
    #plt.plot(x_dense, y_cubic, color='green')

graphs = [(plot_func, 'f(x) = x^2 + 3*ln(x+4)'), (plot_linear_spline, 'Линейный сплайн'), (plot_cubic_spline, 'Кубический сплайн')];

if __name__ == "__main__": 
    print("Расчет погрешностей: ")
    for graph, title in graphs:
        plt.figure(figsize=(10, 6))
        # plt.scatter(x, y, color='red', label='Точки интерполяции')
        plt.grid(True)
        plt.title(title);
        graph();
        plt.show();