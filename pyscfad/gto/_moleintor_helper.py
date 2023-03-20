def int1e_get_dr_order(intor):
    fname = intor.replace('_sph', '').replace('_cart', '')
    if fname[-4:-2] == 'dr':
        orders = [int(fname[-2]), int(fname[-1])]
    else:
        orders = [0, 0]
    return orders

def int2e_get_dr_order(intor):
    fname = intor.replace('_sph', '').replace('_cart', '')
    if fname[-6:-4] == 'dr':
        orders = [int(fname[-4]), int(fname[-3]), int(fname[-2]), int(fname[-1])]
    else:
        orders = [0,] * 4
    return orders

def int1e_dr1_name(intor):
    if 'sph' in intor:
        suffix = '_sph'
    elif 'cart' in intor:
        suffix = '_cart'
    else:
        suffix = ''
    fname = intor.replace('_sph', '').replace('_cart', '')

    if fname[-4:-2] == 'dr':
        orders = [int(fname[-2]), int(fname[-1])]
        intor_ip_bra = fname[:-2] + str(orders[0]+1) + str(orders[1]) + suffix
        intor_ip_ket = fname[:-2] + str(orders[0]) + str(orders[1]+1) + suffix
    else:
        intor_ip_bra = fname + '_dr10' + suffix
        intor_ip_ket = fname + '_dr01' + suffix
    return intor_ip_bra, intor_ip_ket

def int2e_dr1_name(intor):
    if 'sph' in intor:
        suffix = '_sph'
    elif 'cart' in intor:
        suffix = '_cart'
    else:
        suffix = ''
    fname = intor.replace('_sph', '').replace('_cart', '')

    if fname[-6:-4] == 'dr':
        orders = int2e_get_dr_order(intor)
        str1 = str(orders[0]+1) + str(orders[1]) + str(orders[2]) + str(orders[3])
        str2 = str(orders[0]) + str(orders[1]+1) + str(orders[2]) + str(orders[3])
        str3 = str(orders[0]) + str(orders[1]) + str(orders[2]+1) + str(orders[3])
        str4 = str(orders[0]) + str(orders[1]) + str(orders[2]) + str(orders[3]+1)
        intor1 = fname[:-4] + str1 + suffix
        intor2 = fname[:-4] + str2 + suffix
        intor3 = fname[:-4] + str3 + suffix
        intor4 = fname[:-4] + str4 + suffix
    else:
        intor1 = fname + '_dr1000' + suffix
        intor2 = fname + '_dr0100' + suffix
        intor3 = fname + '_dr0010' + suffix
        intor4 = fname + '_dr0001' + suffix
    return intor1, intor2, intor3, intor4
