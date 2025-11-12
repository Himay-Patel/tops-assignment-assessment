"""
Main Controller Module for Fruit Store Console Application
This module serves as the entry point and coordinates between different modules
"""
import sys
import os
from manager import FruitManager
from customer import Customer
from file_handler import FileHandler

class FruitStore:
    def __init__(self):
        self.manager = FruitManager()
        self.customer = Customer()
        self.file_handler = FileHandler()
        self.load_initial_data()
    
    def load_initial_data(self):
        """Load initial fruit stock data from file"""
        try:
            self.manager.fruit_stock = self.file_handler.load_data()
        except Exception as e:
            print(f"Warning: Could not load existing data. Starting with empty stock. Error: {e}")
    
    def display_main_menu(self):
        """Display the main menu and handle user role selection"""
        while True:
            print("\n" + "="*50)
            print("WELCOME TO FRUIT MARKET")
            print("="*50)
            print("1) Manager")
            print("2) Customer")
            print("3) Exit")
            print("="*50)
            
            try:
                role = input("Select your Role: ").strip()
                
                if role == '1':
                    self.manager_menu()
                elif role == '2':
                    self.customer_menu()
                elif role == '3':
                    self.file_handler.save_data(self.manager.fruit_stock)
                    print("Thank you for using Fruit Market! Goodbye!")
                    sys.exit(0)
                else:
                    print("Invalid input! Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\n\nProgram interrupted by user. Saving data and exiting...")
                self.file_handler.save_data(self.manager.fruit_stock)
                sys.exit(0)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    
    def manager_menu(self):
        """Handle manager menu operations"""
        while True:
            print("\n" + "-"*30)
            print("Fruit Market Manager")
            print("-"*30)
            print("1) Add Fruit Stock")
            print("2) View Fruit Stock")
            print("3) Update Fruit Stock")
            print("4) Back to Main Menu")
            print("-"*30)
            
            try:
                choice = input("Enter your choice: ").strip()
                
                if choice == '1':
                    self.manager.add_fruit_stock()
                elif choice == '2':
                    self.manager.view_fruit_stock()
                elif choice == '3':
                    self.manager.update_fruit_stock()
                elif choice == '4':
                    # Save data before going back to main menu
                    self.file_handler.save_data(self.manager.fruit_stock)
                    break
                else:
                    print("Invalid choice! Please enter 1, 2, 3, or 4.")
            except Exception as e:
                print(f"Error in manager menu: {e}")
    
    def customer_menu(self):
        """Handle customer menu operations"""
        while True:
            print("\n" + "-"*30)
            print("Fruit Market Customer")
            print("-"*30)
            print("1) View Available Fruits")
            print("2) Purchase Fruits")
            print("3) Back to Main Menu")
            print("-"*30)
            
            try:
                choice = input("Enter your choice: ").strip()
                
                if choice == '1':
                    self.customer.view_available_fruits(self.manager.fruit_stock)
                elif choice == '2':
                    self.customer.purchase_fruits(self.manager.fruit_stock)
                    # Save updated stock after purchase
                    self.file_handler.save_data(self.manager.fruit_stock)
                elif choice == '3':
                    break
                else:
                    print("Invalid choice! Please enter 1, 2, or 3.")
            except Exception as e:
                print(f"Error in customer menu: {e}")

def main():
    """Main function to start the Fruit Store application"""
    try:
        app = FruitStore()
        app.display_main_menu()
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()