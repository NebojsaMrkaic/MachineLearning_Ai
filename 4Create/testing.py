

def broj_nacina_penjanja(n):
    if n <= 1:
        return 1
    return broj_nacina_penjanja(n-1) + broj_nacina_penjanja(n-2)

print (broj_nacina_penjanja(15))