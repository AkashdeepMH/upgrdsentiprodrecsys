# upgrdsentiprodrecsys
Developed a sentiment-based product recommendation system for Ebuss, an e-commerce platform. Used user reviews and ratings for sentiment analysis and integrated it with collaborative filtering to improve recommendations. Deployed the end-to-end system with Flask and a user-friendly UI.

In today’s rapidly growing e-commerce sector, platforms like Amazon, Flipkart, and Myntra have transformed how consumers purchase goods. Companies no longer need to collect orders physically; customers simply order items directly from websites. I worked as a Machine Learning Engineer at Ebuss, an e-commerce company offering a wide range of products—from books and household essentials to electronics and beauty items.

To help Ebuss gain a competitive edge in this saturated market, I was tasked with designing a sentiment-based product recommendation system that improved product suggestions based on customer reviews and ratings.

To solve this, I followed a structured ML pipeline, which began with exploratory data analysis, data cleaning, and text preprocessing. Various feature extraction techniques were implemented, including TF-IDF and CountVectorizer. I then built and evaluated multiple sentiment classification models—Logistic Regression, Random Forest, and XGBoost—eventually selecting the best-performing model after addressing class imbalance and performing hyperparameter tuning.

Next, I built both user-based and item-based collaborative filtering recommendation systems and selected the one that gave the most relevant results. Based on a selected user's past ratings, the system first recommended 20 likely products. These were further refined by analyzing the sentiment of each product’s reviews using the chosen classification model. From this, the top 5 positively reviewed products were presented to the user.

Finally, I deployed the entire solution using Flask for the backend and Heroku for cloud hosting. A simple HTML interface was created with an input box to enter a username and a submit button. Upon submission, the application displayed the top 5 personalized product recommendations.

All components, including the sentiment model, recommendation engine, HTML frontend, and Flask backend, were integrated and delivered as a complete end-to-end solution.
