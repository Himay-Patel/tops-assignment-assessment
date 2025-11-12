"""
Fruit Manager Module
Handles all manager-related operations including adding, viewing, and updating fruit stock
"""
from file_handler import FileHandler

class FruitManager:
    def __init__(self):
        self.fruit_stock = {}
        self.file_handler = FileHandler()
    
    def add_fruit_stock(self):
        """Add new fruit stock to the inventory"""
        print("\n" + "="*20)
        print("ADD FRUIT STOCK")
        print("="*20)
        
        while True:
            try:
                fruit_name = input("Enter fruit name: ").strip().title()
                
                # Validate fruit name
                if not fruit_name:
                    print("Fruit name cannot be empty!")
                    continue
                
                # Check if fruit already exists
                if fruit_name in self.fruit_stock:
                    print(f"{fruit_name} already exists in stock. Use update option to modify.")
                    break
                
                # Get quantity with validation
                while True:
                    try:
                        quantity = float(input("Enter quantity (in kg): ").strip())
                        if quantity <= 0:
                            print("Quantity must be greater than 0!")
                            continue
                        break
                    except ValueError:
                        print("Invalid quantity! Please enter a valid number.")
                
                # Get price with validation
                while True:
                    try:
                        price = float(input("Enter price per kg: ").strip())
                        if price <= 0:
                            print("Price must be greater than 0!")
                            continue
                        break
                    except ValueError:
                        print("Invalid price! Please enter a valid number.")
                
                # Add fruit to stock
                self.fruit_stock[fruit_name] = {
                    'quantity': quantity,
                    'price': price
                }
                
                # Log the transaction
                log_message = f"Manager added {quantity}kg of {fruit_name} at ₹{price}/kg"
                self.file_handler.log_transaction(log_message)
                
                print(f"\n✅ Successfully added {quantity}kg of {fruit_name} at ₹{price}/kg")
                
                # Ask if user wants to continue
                continue_operation = input("\nDo you want to add more fruits? (y/n): ").strip().lower()
                if continue_operation != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            except Exception as e:
                print(f"Error adding fruit stock: {e}")
    
    def view_fruit_stock(self):
        """Display all available fruit stock"""
        print("\n" + "="*25)
        print("VIEW FRUIT STOCK")
        print("="*25)
        
        if not self.fruit_stock:
            print("No fruits available in stock.")
            return
        
        print("\n{:<15} {:<10} {:<10}".format("Fruit Name", "Quantity", "Price"))
        print("-" * 40)
        
        for fruit, details in self.fruit_stock.items():
            print("{:<15} {:<10} ₹{:<10}".format(
                fruit, 
                f"{details['quantity']}kg", 
                details['price']
            ))
    
    def update_fruit_stock(self):
        """Update existing fruit stock"""
        print("\n" + "="*25)
        print("UPDATE FRUIT STOCK")
        print("="*25)
        
        if not self.fruit_stock:
            print("No fruits available to update.")
            return
        
        # Display current stock for reference
        self.view_fruit_stock()
        
        while True:
            try:
                fruit_name = input("\nEnter fruit name to update: ").strip().title()
                
                if fruit_name not in self.fruit_stock:
                    print(f"{fruit_name} not found in stock!")
                    continue_operation = input("Do you want to try another fruit? (y/n): ").strip().lower()
                    if continue_operation != 'y':
                        break
                    continue
                
                print(f"\nCurrent details for {fruit_name}:")
                print(f"Quantity: {self.fruit_stock[fruit_name]['quantity']}kg")
                print(f"Price: ₹{self.fruit_stock[fruit_name]['price']}/kg")
                
                # Update quantity
                while True:
                    try:
                        new_quantity = float(input("Enter new quantity (in kg): ").strip())
                        if new_quantity < 0:
                            print("Quantity cannot be negative!")
                            continue
                        break
                    except ValueError:
                        print("Invalid quantity! Please enter a valid number.")
                
                # Update price
                while True:
                    try:
                        new_price = float(input("Enter new price per kg: ").strip())
                        if new_price < 0:
                            print("Price cannot be negative!")
                            continue
                        break
                    except ValueError:
                        print("Invalid price! Please enter a valid number.")
                
                old_quantity = self.fruit_stock[fruit_name]['quantity']
                old_price = self.fruit_stock[fruit_name]['price']
                
                # Update the stock
                self.fruit_stock[fruit_name]['quantity'] = new_quantity
                self.fruit_stock[fruit_name]['price'] = new_price
                
                # Log the transaction
                log_message = f"Manager updated {fruit_name}: Quantity {old_quantity}kg→{new_quantity}kg, Price ₹{old_price}→₹{new_price}"
                self.file_handler.log_transaction(log_message)
                
                print(f"\n✅ Successfully updated {fruit_name}")
                print(f"New quantity: {new_quantity}kg")
                print(f"New price: ₹{new_price}/kg")
                
                break
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            except Exception as e:
                print(f"Error updating fruit stock: {e}")