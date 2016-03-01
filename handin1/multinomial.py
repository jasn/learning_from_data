import main as moo
import shared


def main():
    lambdas = [10**i for i in range(-6, 2)] + [0]
    data = shared.load_full_data()

    X, y = data['images'], data['labels']

    thetas = []
    for l in lambdas:
        print("Lambda = %.0e" % l)
        thetas.append({'theta': moo.soft_run(X, y, l),
                       'lambda': l})
        shared.save_data("trained_multinomial.npz", trained=thetas)

if __name__ == "__main__":
    main()
