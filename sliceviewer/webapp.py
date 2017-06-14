from flask_app import app as application

def main():
	'''Entry point for flask app using Werkzeug'''
	from werkzeug.serving import run_simple

	application.config.update({
		})

	run_simple('127.0.0.1', 5000, application,
			use_reloader=True,
			threaded=True)

if __name__ == '__main__':
	main()

# vim: ft=python
