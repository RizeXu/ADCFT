def print_state(N, enr, psi):
    header = f"{'Eigenvalue':^22} | " + " | ".join([f"{f'|{0.5 * (i - 1)}>':^15}" for i in range(N, -N + 1, -1)])
    print(header)
    print("-" * len(header))

    for i in range(N):
        eigenvalue = enr[i] - enr[0]
        eigenvector = psi[:, i]

        ev_str = f"{eigenvalue:.4f}"

        vec_components = []
        for comp in eigenvector:
            comp_str = f"{comp.real:+.4f}{comp.imag:+.4f}j"
            vec_components.append(comp_str)

        vec_str = " | ".join(vec_components)

        print(f"{ev_str:^22} | {vec_str}")