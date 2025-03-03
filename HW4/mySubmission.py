class E_Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

def add(x1, y1, x2, y2, p):
    s = ((y2 - y1) * pow(x2 - x1, -1, p)) % p
    x3 = (s**2-x1-x2) % p
    y3 = (s*(x1-x3)-y1) % p
    return x3, y3

def double(x1, y1, p):
    s = (3 * (x1**2)) * pow(2 * y1, -1, p) % p
    x3 = (s ** 2 - x1 - x1) % p
    y3 = (s*(x1 - x3) - y1) % p
    return x3, y3

def d_and_a(n, point):
    """Calculate n * point using the Double-and-Add algorithm."""

    """ Your code here """
    x = point.x()
    y = point.y()
    curve = point.curve()  # 获取点所属的曲线
    p = curve.p()
    binary = format(n, 'b')  # 將 n 轉換為二進位字串，無 '0b' 前綴
    binary = binary[1:]  # 從索引 1 開始切片
    x1 = x
    y1 = y
    for i in binary:
        if i == '1':
            x1, y1 = double(x1, y1, p)
            x1, y1 = add(x1, y1, x, y, p)
        elif i == '0':
            x1, y1 = double(x1, y1, p)
    result = E_Point(x1, y1)
    return result

#############################################################
# Problem 0: Find base point
def GetCurveParameters():
    # Certicom secp256-k1
    # Hints: https://en.bitcoin.it/wiki/Secp256k1
    _p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    _a = 0x0000000000000000000000000000000000000000000000000000000000000000
    _b = 0x0000000000000000000000000000000000000000000000000000000000000007
    _Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    _Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    _Gz = 0x0000000000000000000000000000000000000000000000000000000000000001
    _n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    _h = 0x01
    return _p, _a, _b, _Gx, _Gy, _Gz, _n, _h


#############################################################
# Problem 1: Evaluate 4G
def compute4G(G, callback_get_INFINITY):
    """Compute 4G"""

    """ Your code here """
    x = G.x()
    y = G.y()
    curve = G.curve()  # 获取点所属的曲线
    p = curve.p()
    x1, y1 = double(x,y,p)
    x2, y2 = double(x1,y1,p)
    #result = ellipticcurve.PointJacobi(G.curve(), hex(x2), hex(y2), G.z(),G.n(), generator=True)
    result = E_Point(x2, y2)
    #result = callback_get_INFINITY()
    return result


#############################################################
# Problem 2: Evaluate 5G
def compute5G(G, callback_get_INFINITY):
    """Compute 5G"""

    """ Your code here """
    x = G.x()
    y = G.y()
    curve = G.curve()  # 获取点所属的曲线
    p = curve.p()
    x1, y1 = double(x, y, p)
    x2, y2 = double(x1, y1, p)
    x3, y3 = add(x2, y2, x, y, p)
    result = E_Point(x3, y3)
    #result = callback_get_INFINITY()

    return result


#############################################################
# Problem 3: Evaluate dG
# Problem 4: Double-and-Add algorithm
def double_and_add(n, point, callback_get_INFINITY):
    """Calculate n * point using the Double-and-Add algorithm."""

    """ Your code here """
    x = point.x()
    y = point.y()
    curve = point.curve()  # 获取点所属的曲线
    p = curve.p()
    binary = format(n, 'b')  # 將 n 轉換為二進位字串，無 '0b' 前綴
    binary = binary[1:]  # 從索引 1 開始切片
    x1 = x
    y1 = y
    num_doubles = 0
    num_additions = 0
    for i in binary:
        if i == '1':
            num_doubles += 1
            num_additions += 1
            x1, y1 = double(x1, y1, p)
            x1, y1 = add(x1, y1, x, y, p)
        elif i == '0':
            num_doubles += 1
            x1, y1 = double(x1, y1, p)


    result = E_Point(x1, y1)


    return result, num_doubles, num_additions


#############################################################
# Problem 5: Optimized Double-and-Add algorithm
def optimized_double_and_add(n, point, callback_get_INFINITY):
    """Optimized Double-and-Add algorithm that simplifies sequences of consecutive 1's."""

    """ Your code here """
    result = callback_get_INFINITY()
    result = d_and_a(n, point)
    num_doubles = 0
    num_additions = 0

    # 找到 n 所屬的 2 的次方區間
    upper_bound = 1 << n.bit_length()  # 最接近且 > n 的 2^(k+1)
    ub_power = n.bit_length()
    # 方法 A: 加法方式
    remaining = n
    add_count = 0
    consecutive_half_count = 0
    last_power = None

    while remaining > 0:
        power = 1 << (remaining.bit_length() - 1)
        remaining -= power

        # 判斷是否與上一個 power 相差兩倍
        if last_power is not None and power == last_power // 2:
            consecutive_half_count += 1
        else:
            consecutive_half_count = 1  # 重置連續次數

        # 如果連續相差兩倍超過 2 個，從第三個開始不增加 add_count
        if consecutive_half_count <= 2:
            add_count += 1

        # 更新上一個 power
        last_power = power

    # 方法 B: 減法方式
    remaining = upper_bound - n
    subtract_count = 0
    while remaining > 0:
        power = 1 << (remaining.bit_length() - 1)
        remaining -= power
        subtract_count += 1
    add_count -= 1
    # print(add_count)
    # print(subtract_count)

    if subtract_count < add_count:
        num_doubles = ub_power
        num_additions = subtract_count
        return result, num_doubles, num_additions
    else:
        num_doubles = ub_power - 1
        num_additions = add_count
        return result, num_doubles, num_additions


#############################################################
# Problem 6: Sign a Bitcoin transaction with a random k and private key d
def sign_transaction(private_key, hashID, callback_getG, callback_get_n, callback_randint):
    """Sign a bitcoin transaction using the private key."""

    """ Your code here """
    G = callback_getG()
    n = callback_get_n()
    da = private_key
    z = int(hashID[:n], 16)

    while True:
        k = callback_randint(1, n - 1)
        point = d_and_a(k, G)
        x = point.x()
        r = x % n
        if r == 0:
            continue

        s = (pow(k, -1, n) * (z + r * da)) % n
        if s != 0:
            return [r, s]

    #return signature


##############################################################
# Step 7: Verify the digital signature with the public key Q
def verify_signature(public_key, hashID, signature, callback_getG, callback_get_n, callback_get_INFINITY):
    """Verify the digital signature."""

    """ Your code here """
    G = callback_getG()
    curve = G.curve()  # 获取点所属的曲线
    p = curve.p()
    b = curve.b()
    n = callback_get_n()
    infinity_point = callback_get_INFINITY()
    if public_key == infinity_point or (public_key.y()**2) % p != (public_key.x()**3 + b) % p or n*public_key != infinity_point:
        return False

    r = signature[0]
    s = signature[1]
    if not (1 <= r <= n - 1 and 1 <= s <= n - 1):
        return False

    z = int(hashID[:n], 16)
    w = pow(s, -1, n)
    u1 = z * w % n
    u2 = r * w % n
    p1 = d_and_a(u1, G)
    p2 = d_and_a(u2, public_key)
    x1, y1 = add(p1.x(), p1.y(), p2.x(), p2.y(), p)
    if r % n == x1 % n:
        is_valid_signature = True
    else:
        is_valid_signature = False


    #is_valid_signature = True if callback_get_n() > 0 else False

    return is_valid_signature
