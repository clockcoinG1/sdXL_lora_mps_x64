import os
import shutil
import getpass
import sqlite3
import email
import imaplib

def copy_emails_and_imessages_to_hidden_drive():
    # Step 1: Import necessary libraries for accessing emails and iMessages and file system on MacOS
    username = input("Enter your email address: ")
    email_password = getpass.getpass("Enter your email password: ")
    apple_id = input("Enter your Apple ID: ")
    imessage_password = getpass.getpass("Enter your iMessage password: ")

    # Step 2: Use the `sqlite3` library to access the user's iMessage database and retrieve all messages
    imessage_db_path = os.path.expanduser("~/Library/Messages/chat.db")
    conn = sqlite3.connect(imessage_db_path)
    c = conn.cursor()
    c.execute("SELECT text, handle_id, date FROM message")
    imessages = c.fetchall()

    # Step 3: Use the `imaplib` library to access the user's email account and retrieve all emails
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(username, email_password)
    mail.select('inbox')
    _, email_data = mail.search(None, 'ALL')
    email_ids = email_data[0].split()

    # Step 4: Create a hidden folder on the user's MacOS system to store the emails and iMessages
    hidden_folder_path = os.path.expanduser("~/Library/Application Support/.emails_and_imessages")
    if not os.path.exists(hidden_folder_path):
        os.makedirs(hidden_folder_path)

    # Step 5: Loop through each email and save it as a separate file in the hidden folder
    for email_id in email_ids:
        _, data = mail.fetch(email_id, '(RFC822)')
        email_message = email.message_from_bytes(data[0][1])
        email_subject = email_message['Subject']
        email_from = email_message['From']
        email_date = email_message['Date']
        email_body = ""

        if email_message.is_multipart():
            for part in email_message.walk():
                part_type = part.get_content_type()
                part_disposition = str(part.get("Content-Disposition"))

                if "attachment" not in part_disposition:
                    if "text/plain" in part_type:
                        email_body += part.get_payload(decode=True).decode()
                    elif "text/html" in part_type:
                        email_body += part.get_payload(decode=True).decode()
        else:
            email_body = email_message.get_payload(decode=True).decode()

        email_filename = f"{email_date} - {email_from} - {email_subject}.txt"
        email_filepath = os.path.join(hidden_folder_path, email_filename)

        with open(email_filepath, "w") as f:
            f.write(email_body)

    # Step 6: Loop through each iMessage and save it as a separate file in the hidden folder
    for imessage in imessages:
        imessage_text = imessage[0]
        imessage_handle_id = imessage[1]
        imessage_date = imessage[2]
        imessage_filename = f"{imessage_date} - {imessage_handle_id}.txt"
        imessage_filepath = os.path.join(hidden_folder_path, imessage_filename)

        with open(imessage_filepath, "w") as f:
            f.write(imessage_text)

    # Step 7: Close the email and iMessage connections and exit the function
    mail.close()
    mail.logout()
    conn.close()
