import socket

# Configuración: Tu propio ordenador y el puerto del juego
UDP_IP = "127.0.0.1"
UDP_PORT = 20777

# Creamos el "embudo" para escuchar la red
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"📡 Escuchando telemetría en {UDP_IP}:{UDP_PORT}...")
print("Arranca el F1 24, sal a pista y acelera. Dale a Ctrl+C para parar esto.\n")

try:
    while True:
        # Esperamos a recibir un paquete (buffer de 2048 bytes)
        data, addr = sock.recvfrom(2048) 
        print(f"✅ ¡Paquete recibido! Origen: {addr} | Tamaño: {len(data)} bytes")
except KeyboardInterrupt:
    print("\n🛑 Escucha detenida.")