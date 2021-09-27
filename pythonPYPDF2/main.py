
import pyttsx3
import PyPDF2
#book = (A1.pdf, 'rb')
book = (r'C:\Booksri\B.pdf')
pdfReader = PyPDF2.PdfFileReader(book)
pages = pdfReader.numPages
print(pages)
speaker = pyttsx3.init()
for num in range(18, pages):
     page = range
     page =pdfReader.getPage(18)
     text = page.extractText()
     speaker.say(text)
     speaker.runAndWait()






'''import pyttsx3
import PyPDF2
#book = (A1.pdf, 'rb')
book = (r'C:\Sales_Data\A1.pdf')
pdfReader = PyPDF2.PdfFileReader(book)
pages = pdfReader.numPages
print(pages)
speaker = pyttsx3.init()
page = pdfReader.getPage(12)
text = page.extractText()
speaker.say(text)
speaker.runAndWait()

import pyttsx3
import PyPDF2
#book = (A1.pdf, 'rb')
book = (r'C:\Sales_Data\A1.pdf')
pdfReader = PyPDF2.PdfFileReader(book)
pages = pdfReader.numPages
print(pages)
speaker = pyttsx3.init()
for num in range(12, pages):
     page = range
     page =pdfReader.getPage(12)
     text = page.extractText()
     speaker.say(text)
     speaker.runAndWait()'''


