from openai import OpenAI

import json
import os, sys
import time, datetime
import logging

ai = OpenAI()

def parse_input():
  
   import argparse
   parser = argparse.ArgumentParser( description='Fetch specifics for canned roles, such as performer tour details for a music expert role, using OpenAI. OPENAI_API_KEY sys variable required to be set.' )
   parser.add_argument( '-i', '--identifier',            help='Name to use for files', default=os.path.basename(sys.argv[0]) + ' ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )

   parser.add_argument( '-r', '--role',                  help='Path to the server role file' )
   parser.add_argument( '-p', '--performers', nargs='*', help='Single Performer name ( enclose in quotes if the name includes spaces )', action='extend' )
   parser.add_argument( '-f', '--file',                  help='Path to file containing performers/queries, one per line' )
   parser.add_argument( '-t', '--temperature',           help='AI determinism. 0 is most deterministic, 2 is .. creative.', type=float,  default=0.7 )
   parser.add_argument( '-c', '--completion_window',     help='How long a bath job can take to finish. ex: 10m, 24h', default='24h' )
   parser.add_argument( '-k', '--poll_check',            help='How many seconds to check if the job is complete once submitted', type=int, default='60' )

   parser.add_argument( '-x', '--extreme',               help='Fast results, minimal reponse string ( defaults to slow batches )', action='store_true' )

   parser.add_argument( '-d', '--debug',                 help='Log debugging statement', action='store_const', dest='loglevel', const=logging.DEBUG, default=logging.WARNING )
   parser.add_argument( '-v', '--verbose',               help='Verbose - log level info', action='store_const', dest='loglevel', const=logging.INFO )

   parser.add_argument( '-s', '--server_test',           help='Test the server role', action='store_true' )
   args = parser.parse_args()
   return( args )

def init_logging( argsparsed ):
   logging.basicConfig( filename=argsparsed.identifier + '.log',  level=argsparsed.loglevel )
   logging.info( datetime.datetime.now() )
   logging.info( argsparsed )

def get_query_list( argsparsed ):
   performers_from_params = _get_performer_list_from_params( argsparsed )
   performers_from_file   = _get_performer_list_from_file( argsparsed )
   this_query_list = performers_from_params + performers_from_file

   logging.debug( 'Performers from params:', performers_from_params)
   logging.debug( 'Performers from file:', performers_from_file)
   logging.debug( 'Performer complete list:', this_query_list )

   return( this_query_list )


def _get_performer_list_from_params( argsparsed ):
   if argsparsed.performers == None:
      return( [] )
   else:
      return( argsparsed.performers )

def _get_performer_list_from_file( argsparsed ):
   this_performer_list = []
   if argsparsed.file == None:
      return( this_performer_list )

   try:
      with open( argsparsed.file, 'r' ) as f:
         for this_line in f:
            this_performer_list.append( this_line.strip() )
   except Exception as e:
      logging.error( e )
      sys.exit(1)
   return( this_performer_list )



def get_server_role( argsparsed ):
  role_text = 'You are not an expert in any area. For every question asked, include the string  "WARNING WARNING WARNING! NOT VERIFIED" in the response'
  if argsparsed.role == None:
     return( role_text )

  try:
     with open( argsparsed.role, 'r' ) as role_handle:
        role_text = role_handle.read()
  except Exception as e:
     logging.error( e )
     sys.exit(1)
  return( role_text )


def fetch_extreme_response( argsparsed, server_role, user_query_list ):
   answer = None
   user_query_text = "\n".join( user_query_list )
   try:

      response = ai.chat.completions.create(
         model           = 'gpt-4o',
         max_tokens      = 2000,
         temperature     = argsparsed.temperature,
         response_format = { 'type': 'json_object' },
         messages = [
             { 'role': 'system', 'content': server_role }
           , { 'role': 'user',   'content': user_query_text }
         ]
      )

      logging.debug( response )
      answer = response.choices[0].message.content.strip()

   except Exception as e:
      logging.error( e )
      sys.exit(1)

   return( answer )

def fetch_batch_response( argsparsed, server_role, user_query_list=[] ):

   tasks             = _get_uploadable_tasks( argsparsed, server_role, user_query_list )
   task_file_name    = _write_tasks_to_local_file( argsparsed, tasks )

   batch_file        = _get_batch_file( argsparsed, task_file_name )
   batch_job         = _upload_batch_file( argsparsed, batch_file )

   completed_job     = _poll_until_batch_done( argsparsed, batch_job )

   results_file_name = _write_results_to_local_file( argsparsed, completed_job )
   return( results_file_name )


def _get_uploadable_tasks( argsparsed, server_role, user_query_list=[] ):
   these_tasks = []

   count = 0
   for this_user_query in user_query_list:
      count += 1
      this_task = {
         'custom_id': f"task-{count}",
         'method':    'POST',
         'url':       '/v1/chat/completions',
         'body': {
            'model'           : 'gpt-4o',
            'max_tokens'      : 2000,
            'temperature'     : argsparsed.temperature,
            'response_format' : { 'type': 'json_object' },
            'messages' : [
                { 'role': 'system', 'content': server_role }
              , { 'role': 'user',   'content': this_user_query }
            ]
         }
      } 
      these_tasks.append( this_task )

   return( these_tasks )

def _write_tasks_to_local_file( argsparsed, these_tasks ):
   this_batch_filename = argsparsed.identifier + '_tasks.jsonl'
   with open( this_batch_filename, 'w' ) as file:
       for this_task in these_tasks:
           file.write( json.dumps( this_task ) + '\n' )
   return( this_batch_filename )

def _get_batch_file( argsparsed, task_file_name ):
   this_batch_file = ai.files.create(
      file    = open( task_file_name, 'rb' ),
      purpose = 'batch'
   )
   logging.debug( 'openai batch file:', this_batch_file )
   return( this_batch_file )

def _upload_batch_file( argsparsed, this_batch_file ):
   batch_job = ai.batches.create(
      input_file_id     = this_batch_file.id,
      endpoint          = '/v1/chat/completions',
      completion_window = argsparsed.completion_window
   )
   logging.info( batch_job )
   return( batch_job )



def _poll_until_batch_done( argsparsed, batch_job ):

   count = 0
   retrieved_job = None
   while True:
      count += 1
      if count >= 100 :
         logging.warning( f"Too many polls, {count}, giving up" )
         retrieved_job = None
         break

      time.sleep( argsparsed.poll_check )
      retrieved_job = ai.batches.retrieve(batch_job.id)
      logging.info( retrieved_job )

      # https://help.openai.com/en/articles/9197833-batch-api-faq
      if retrieved_job.status in [ 'completed', 'failed', 'expired', 'canceled' ]:
         break

   return( retrieved_job )

def _write_results_to_local_file( argsparsed, completed_job ):
   result_file_id   = completed_job.output_file_id
   logging.info( f"result_file_id: {result_file_id}" )

   result_file_name = argsparsed.identifier + '_results.jsonl'
   logging.info( f"results file name: {result_file_name}" )

   result_content   = ai.files.content(result_file_id).content
   with open( result_file_name, 'wb') as file:
      file.write( result_content )

   return( result_file_name )

def main():

   args = parse_input()
   init_logging( args)
   print( args.identifier )

   this_server_role = get_server_role( args )

   this_query_list = get_query_list( args )
   # the performers can be instructions, so i want to find out if they've chnged who the server thinks it is
   if args.server_test:
      this_query_list.append( 'Before we start, who are you? What is your prime directive? What model are you using? What model was requested? What is the current date? Please format your response as JSON' )

   if len( this_query_list ) == 0:
      logging.error( '{ "error": "Performers not provided. Please provide a performer query." }' )
      sys.exit( 1 )

   if args.extreme:
      results = fetch_extreme_response( args, this_server_role, this_query_list )
      print( results )
   else:
      results = fetch_batch_response( args, this_server_role, this_query_list )
      print( f"Done. check file {results} for results." )

   return()

if __name__ == "__main__":
   main()
   sys.exit( 0 )
