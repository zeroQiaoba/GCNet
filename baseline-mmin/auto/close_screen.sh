set -e
grep_name=$1
echo "screen contains name $grep_name:"
screen -ls | grep $grep_name
while true
do
	read -r -p "Close these screens? [Y/n] " input

	case $input in
	    [yY][eE][sS]|[yY])
			screen -ls | awk '{print $1}'| grep $grep_name | awk '{print "screen -S "$1" -X quit"}'| sh
			echo "Finished"
            exit 1
			;;

	    [nN][oO]|[nN])
			echo "Abort"
			exit 1	       	
			;;

	    *)
			echo "Invalid input..."
			;;
	esac
done
