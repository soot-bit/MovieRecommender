from src.trainer import Trainer
from src.py.vanilla.als import ALSModel


def main():
    trainer = Trainer()

    epochs = 20
    model = trainer.fit(ALSModel, epochs)

    user_id = 1  
    movie_id = 10 
    predicted_rating = model.predict(user_id, movie_id)

    print(f"Predicted rating for User {user_id} on Movie {movie_id}: {predicted_rating:.2f}")

if __name__ == '__main__':
    main()
