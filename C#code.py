def embed_info(data, x, y, N, flow, count):
    tee = y
    for _ in range(N):
        if count < flow.size:  # Use .size for numpy arrays
            data[x, y] = complex(flow[count].real, flow[count].imag)  # Use .real and .imag
            y += 1
            count += 1
        else:
            break
    
    xss = x
    for _ in range(N):
        if count < flow.size:  # Use .size for numpy arrays
            data[x, y] = complex(flow[count].real, flow[count].imag)  # Use .real and .imag
            x += 1
            count += 1
        else:
            break

    tee = y
    for _ in range(N):
        if count < flow.size:  # Use .size for numpy arrays
            data[x, y] = complex(flow[count].real, flow[count].imag)  # Use .real and .imag
            y -= 1
            count += 1
        else:
            break

    xss = x
    for _ in range(N):
        if count < flow.size:  # Use .size for numpy arrays
            data[x, y] = complex(flow[count].real, flow[count].imag)  # Use .real and .imag
            x -= 1
            count += 1
        else:
            break

    Curren_X = x + 1
    Curren_Y = y + 1
    if count == flow.size:  # Use .size for numpy arrays
        return data
    else:
        N -= 2
        return embed_info(data, Curren_X, Curren_Y, N, flow, count)
