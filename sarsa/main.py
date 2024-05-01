from training import train
from simulation import run_simulation
from file_operations import load_policy
from config_and_helpers import Q, n_bins, nA, env

def main():
    print("Welcome to the Lunar Lander simulation!")
    choice = input("Do you want to train a new model or load an existing one? Enter 'train' or 'load': ").strip().lower()

    if choice == 'train':
        Q[:] = train()  # This function will also save the model and hyperparameters at the end of training
    elif choice == 'load':
        filename = input("Please enter the filename of the saved model: ")
        loaded_data = load_policy(filename)
        Q[:] = loaded_data['Q_table']  # Update the Q-table with the loaded policy
    else:
        print("Invalid input. Exiting.")
        return

    run_simulation()

if __name__ == "__main__":
    main()
