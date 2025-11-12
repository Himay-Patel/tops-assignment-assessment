"""
Customer Module
Handles all customer-related operations including viewing available fruits and purchasing
"""
from file_handler import FileHandler

class Customer:
    def __init__(self):
        self.file_handler = FileHandler()
    
    def view_available_fruits(self, fruit_stock):
        """Display all available fruits for customer"""
        print("\n" + "="*30)
        print("AVAILABLE FRUITS")
        print("="*30)
        
        if not fruit_stock:
            print("No fruits available at the moment.")
            return
        
        print("\n{:<15} {:<10} {:<10}".format("Fruit Name", "Quantity", "Price/kg"))
        print("-" * 45)
        
        for fruit, details in fruit_stock.items():
            if details['quantity'] > 0:
                print("{:<15} {:<10} ₹{:<10}".format(
                    fruit, 
                    f"{details['quantity']}kg", 
                    details['price']
                ))
            else:
                print("{:<15} {:<10} {:<10}".format(
                    fruit, 
                    "Out of Stock", 
                    f"₹{details['price']}"
                ))
    
    def purchase_fruits(self, fruit_stock):
        """Handle fruit purchase by customer"""
        print("\n" + "="*20)
        print("PURCHASE FRUITS")
        print("="*20)
        
        if not fruit_stock:
            print("No fruits available for purchase.")
            return
        
        # Display available fruits
        self.view_available_fruits(fruit_stock)
        
        cart = {}
        total_amount = 0
        
        while True:
            try:
                fruit_name = input("\nEnter fruit name to purchase (or 'done' to finish): ").strip().title()
                
                if fruit_name.lower() == 'done':
                    break
                
                if fruit_name not in fruit_stock:
                    print(f"{fruit_name} not available!")
                    continue
                
                if fruit_stock[fruit_name]['quantity'] <= 0:
                    print(f"Sorry, {fruit_name} is out of stock!")
                    continue
                
                available_quantity = fruit_stock[fruit_name]['quantity']
                price_per_kg = fruit_stock[fruit_name]['price']
                
                print(f"Available quantity: {available_quantity}kg")
                print(f"Price: ₹{price_per_kg}/kg")
                
                # Get purchase quantity with validation
                while True:
                    try:
                        purchase_quantity = float(input("Enter quantity to purchase (in kg): ").strip())
                        if purchase_quantity <= 0:
                            print("Quantity must be greater than 0!")
                            continue
                        if purchase_quantity > available_quantity:
                            print(f"Not enough stock! Only {available_quantity}kg available.")
                            continue
                        break
                    except ValueError:
                        print("Invalid quantity! Please enter a valid number.")
                
                # Calculate cost for this fruit
                fruit_cost = purchase_quantity * price_per_kg
                
                # Add to cart
                if fruit_name in cart:
                    cart[fruit_name]['quantity'] += purchase_quantity
                    cart[fruit_name]['cost'] += fruit_cost
                else:
                    cart[fruit_name] = {
                        'quantity': purchase_quantity,
                        'cost': fruit_cost,
                        'price_per_kg': price_per_kg
                    }
                
                total_amount += fruit_cost
                
                print(f"Added {purchase_quantity}kg of {fruit_name} to cart - ₹{fruit_cost}")
                
                continue_shopping = input("\nDo you want to add more fruits? (y/n): ").strip().lower()
                if continue_shopping != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\nPurchase cancelled by user.")
                return
            except Exception as e:
                print(f"Error during purchase: {e}")
        
        # Process the purchase if cart is not empty
        if cart:
            self._process_purchase(cart, fruit_stock, total_amount)
        else:
            print("No items in cart. Purchase cancelled.")
    
    def _process_purchase(self, cart, fruit_stock, total_amount):
        """Process the final purchase and update stock"""
        print("\n" + "="*25)
        print("PURCHASE SUMMARY")
        print("="*25)
        
        print("\n{:<15} {:<10} {:<10} {:<10}".format("Fruit", "Quantity", "Price/kg", "Total"))
        print("-" * 55)
        
        for fruit, details in cart.items():
            print("{:<15} {:<10} ₹{:<10} ₹{:<10}".format(
                fruit,
                f"{details['quantity']}kg",
                details['price_per_kg'],
                details['cost']
            ))
        
        print("-" * 55)
        print(f"TOTAL AMOUNT: ₹{total_amount}")
        
        # Confirm purchase
        confirm = input("\nConfirm purchase? (y/n): ").strip().lower()
        
        if confirm == 'y':
            # Update stock and log transaction
            for fruit, details in cart.items():
                fruit_stock[fruit]['quantity'] -= details['quantity']
                
                # Log each fruit purchase
                log_message = f"Customer purchased {details['quantity']}kg of {fruit} for ₹{details['cost']}"
                self.file_handler.log_transaction(log_message)
            
            # Log total transaction
            total_log_message = f"Total purchase amount: ₹{total_amount}"
            self.file_handler.log_transaction(total_log_message)
            
            print(f"\n✅ Purchase successful! Total amount: ₹{total_amount}")
            print("Thank you for your purchase!")
        else:
            print("Purchase cancelled.")