# basics of pydantic

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    # can set default value
    name: str = 'Yash'
    # set integer as optional datatype, it will convert string into integers
    age: Optional[int] = None
    # email validator, validate only genuine emails
    email: EmailStr = 'yash@abc.com'
    # we can set range, default value and description 
    cgpa:float = Field(gt=5, lt=10, default=7, description='This is cgpa of a student')

new_student = {'name': 'Yash', 'age': '32'}

student = Student(**new_student)

# student we are getting here is a pydantic object 
# we can convert it into dictionary, json 
print(student)

# converting to dictionary
print(dict(student))

# converting to json
print(student.model_dump_json())