def get_customer_by_id(customer_id: int):
    session = SessionLocal()
    customer = session.query(Customer).filter(Customer.customer_id == customer_id).first()
    session.close()
    return customer
