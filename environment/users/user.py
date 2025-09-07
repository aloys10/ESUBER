class User:
    """
    Object to represent users that will interacts with the s
    """

    id_counter = 0

    def __init__(self, name, gender, age, description, job="", hobby="", activity_level=2, conformity_level=2, diversity_level=2):
        """
        id (integer): unique identifier for each user
        name (string): name + surname of the user
        gender (string): 'M' for male and 'F' for female
        age (integer): the age of the user between 0 and 200
        description (string): small description of the user, including its cinematic interests
        job (string, optional): the job of the user
        hobby (string, optional): the hobby of the user
        activity_level (integer, optional): activity level (1-3) for emotional rating criteria
        conformity_level (integer, optional): conformity level (1-3) for emotional rating criteria
        diversity_level (integer, optional): diversity level (1-3) for emotional rating criteria
        """
        self.id = User.id_counter
        User.id_counter += 1
        self.name = name
        self.gender = gender
        self.age = age
        self.description = description
        self.job = job
        self.hobby = hobby
        self.activity_level = activity_level
        self.conformity_level = conformity_level
        self.diversity_level = diversity_level

    def __str__(self) -> str:
        return (
            f"User(id = {self.id}, name = {self.name}, gender = {self.gender}, age ="
            f" {self.age}:\n{self.description}), job = {self.job}, hobby ="
            f" {self.hobby}, activity_level = {self.activity_level}, conformity_level = {self.conformity_level}, diversity_level = {self.diversity_level})"
        )

    def __repr__(self) -> str:
        return f"User(id = {self.id}, name = {self.name})"

    @staticmethod
    def get_num_users():
        return User.id_counter
