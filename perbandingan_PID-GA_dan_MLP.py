import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# parameter sistem motor
J = 0.01  #moment of inertia
b = 0.1   #damping ratio
K = 0.01  #motor constant
setpoint = 1000  # Target kecepatan 

#setting waktu
t = np.linspace(0, 10, 1000)
dt = t[1] - t[0]

def motor_speed(x, t, u):
    theta, omega = x
    dxdt = [omega, (u - b*omega - K*theta)/J]
    return dxdt

def pid_controller(y, setpoint, integral, last_error, dt, Kp, Ki, Kd):
    error = setpoint - y
    integral += error * dt
    derivative = (error - last_error) / dt
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral, error

def run_simulation(Kp, Ki, Kd):
    """Fungsi untuk menjalankan simulasi motor speed"""
    y0 = [0, 0]
    y = [0]
    integral_err = 0
    last_err = 0
    for i in range(1, len(t)):
        u, integral, err = pid_controller(y[-1], setpoint, integral_err, last_err, dt, Kp, Ki, Kd)
        integral_err = integral
        last_err = err
        next_point = odeint(motor_speed, y0, [t[i-1], t[i]], args=(u,))[-1]
        y0 = next_point
        y.append(next_point[0])
    return np.array(y)


# PERSIAPAN DATA & TRAINING Multi-Layer Perceptron (MLP)

print("Membaca data hasil Genetic Algorithm dari Excel...")
df = pd.read_excel('pid_ga_optimization_records.xlsx')
df = df.dropna() # Membersihkan data

# Cari PID terbaik dari Genetic Algorithm
best_ga_row = df.loc[df['Fitness'].idxmin()]
ga_Kp, ga_Ki, ga_Kd = best_ga_row['Kp'], best_ga_row['Ki'], best_ga_row['Kd']

print("Melatih MLP...")
# Input: Overshoot, RiseTime, SettlingTime, RMSE 
X = df[['Overshoot', 'RiseTime', 'SettlingTime', 'RMSE']].values
# Output: Kp, Ki, Kd
y = df[['Kp', 'Ki', 'Kd']].values

# Standarisasi data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Membuat dan melatih model Multi-Layer Perceptron (MLP)
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=2500, random_state=42)
mlp.fit(X_scaled, y_scaled)


# Prediksi PID menggunakan Multi-Layer Perceptron (MLP)

# Kondisi super ideal: Overshoot=0, RiseTime=0, SettlingTime=0, RMSE=0
kondisi_ideal = np.array([[0, 0, 0, 0]])
kondisi_ideal_scaled = scaler_X.transform(kondisi_ideal)

prediksi_pid_scaled = mlp.predict(kondisi_ideal_scaled)
mlp_Kp, mlp_Ki, mlp_Kd = scaler_y.inverse_transform(prediksi_pid_scaled)[0]

# Memastikan nilai PID tidak negatif
mlp_Kp, mlp_Ki, mlp_Kd = max(0, mlp_Kp), max(0, mlp_Ki), max(0, mlp_Kd)


# SIMULASI & GRAFIK

print("Menjalankan simulasi perbandingan PID-GA dan PID-MLP...")
y_ga = run_simulation(ga_Kp, ga_Ki, ga_Kd)
y_mlp = run_simulation(mlp_Kp, mlp_Ki, mlp_Kd)

print("\n=== HASIL NILAI PID ===")
print(f"GA  -> Kp: {ga_Kp:.4f}, Ki: {ga_Ki:.4f}, Kd: {ga_Kd:.4f}")
print(f"MLP -> Kp: {mlp_Kp:.4f}, Ki: {mlp_Ki:.4f}, Kd: {mlp_Kd:.4f}")

# Plot Grafik
plt.figure(figsize=(10, 6))
plt.plot(t, y_ga, label=f'PID-GA (Biru)', color='blue', linewidth=2)
plt.plot(t, y_mlp, label=f'PID-MLP (Merah putus-putus)', color='red', linestyle='--', linewidth=2)
plt.axhline(setpoint, color='green', linestyle=':', label='Target (1000 RPM)')

plt.title('Perbandingan Respon Motor Speed: PID-GA vs PID-MLP')
plt.xlabel('Waktu (detik)')
plt.ylabel('Motor Speed (RPM)')
plt.legend(loc='lower right')
plt.grid(True)

# Ambil riwayat error selama MLP belajar
loss_history = mlp.loss_curve_

# Simpan ke dalam file excel baru
df_mlp_record = pd.DataFrame({
    'Epoch (Iterasi Belajar)': range(1, len(loss_history) + 1),
    'Loss (Tingkat Error)': loss_history
})
df_mlp_record.to_excel('mlp_training_record.xlsx', index=False)
print("Catatan belajar MLP berhasil disimpan ke 'mlp_training_record.xlsx'")

# Gambar kurva belajar MLP di jendela terpisah
plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='purple', linewidth=2)
plt.title('Kurva Pembelajaran (Loss Curve) PID-MLP')
plt.xlabel('Epoch (Iterasi Belajar)')
plt.ylabel('Loss (Tingkat Error)')
plt.grid(True)
plt.show()

