
# Food Delivery Platform

Welcome to the Food Delivery Platform! This project features an NLP-powered chatbot for intuitive ordering, integrated with Flask and MongoDB for seamless web application functionality and data management.
![Food Delivery Platform](static/images/Screenshot)
## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This platform is designed to provide a seamless and user-friendly food ordering experience. The core features include an NLP-powered chatbot that helps users place orders intuitively, a robust backend powered by Flask, and MongoDB for efficient data management.

## Features

- **NLP-powered Chatbot**: Helps users place orders through natural language processing.
- **Flask Integration**: Powers the web application, handling requests and responses.
- **MongoDB Integration**: Efficiently manages data related to users, orders, and menus.
- **Responsive Design**: Ensures a smooth user experience across different devices.
- **Secure Authentication**: Protects user data and ensures secure transactions.

## Installation

### Prerequisites

- Python 3.8+
- MongoDB
- Flask

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Raresanju/Rare-Foods.git
   cd food-delivery-platform
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MongoDB**:
   - Ensure MongoDB is running on your machine or a server.
   - Create a database named `food_delivery`.

5. **Configure environment variables**:
   - Create a `.env` file in the project root directory and add the following:
     ```env
     FLASK_APP=app.py
     FLASK_ENV=development
     MONGO_URI=mongodb://localhost:27017/food_delivery
     SECRET_KEY=your_secret_key
     ```

6. **Run the application**:
   ```bash
   flask run
   ```

## Usage

- Open your web browser and go to `http://localhost:5000` to access the platform.
- Interact with the NLP-powered chatbot to place orders.
- Use the admin panel to manage menus, orders, and user data.

## Contributing

We welcome contributions to enhance the platform. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a pull request.

## License

This project is licensed under the MIT License. 
Feel free to customize the sections and content according to your specific project details and preferences.
