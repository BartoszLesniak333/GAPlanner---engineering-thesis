from vrptw.data import load_instance

inst = load_instance("data/data1.csv")  

print("⚙️  Wczytano instancję")
print("• Wiersze (z depotem):", len(inst.data))
print("• Klienci:", len(inst.data) - 1)
print("• Q:", inst.Q)
print("• Kolumny:", list(inst.data.columns))
print(inst.data.head(3))               
print("• Macierz dist shape:", inst.distance.shape)
print(inst.distance[:6,:6])  